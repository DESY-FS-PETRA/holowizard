import cupy as cp
import cupyx.scipy.fft as cufft
import pytest
import scipy.fft

scipy.fft.set_global_backend(cufft)


@pytest.fixture
def callbacks():

    data_shape = (256, 256)
    code = r"""
                __device__ cufftComplex CB_ConvertInputC(
                void *dataIn,
                size_t offset,
                void *callerInfo,
                void *sharedPtr)
                {
                cufftComplex x;
                x.x = 1.;
                x.y = 0.;
                return x;
                }
            __device__ cufftCallbackLoadC d_loadCallbackPtr = CB_ConvertInputC;
            """

    # https: // docs.nvidia.com / cuda / cufft / index.html  # cufft-callback-routines

    backwardCallback = r"""
                #define DATASIZE {arrsize}
                __device__ __host__ inline
                cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
                {{
                cufftComplex c;
                c.x = a.x * b.x - a.y * b.y;
                c.y = a.x * b.y + a.y * b.x;
                return c;
                }}

                __device__ void FresnelCallback(
                void *dataOut,
                size_t offset,
                cufftComplex element,
                void *callerInfo,
                void *sharedPtr)
                {{
                cufftComplex *kernel = (cufftComplex*)callerInfo;
                cufftComplex kernelEl = kernel[offset];
                kernelEl.x *= 1./DATASIZE;
                kernelEl.y *=-1./DATASIZE;

                cufftComplex value = ComplexMul(element, kernelEl);

                ((cufftComplex*)dataOut)[offset] = value;
                }}
                 __device__ cufftCallbackStoreC d_storeCallbackPtr = FresnelCallback;      
                """.format(
        arrsize=data_shape[0] * data_shape[1]
    )

    yield data_shape, backwardCallback


def test_backward_single_callback(callbacks):

    data_shape, backwardCallback = callbacks

    x = cp.zeros(data_shape, dtype=cp.complex64)
    kernel = cp.ones(data_shape, dtype=cp.complex64)
    with cp.fft.config.set_cufft_callbacks(
        cb_store=backwardCallback, cb_store_aux_arr=kernel
    ):
        y = cp.fft.fft2(cp.from_dlpack(x))  # cp.fft.fft2(cp.from_dlpack(x))

    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_backward_double_vanilla(callbacks):

    data_shape, backwardCallback = callbacks

    x = cp.zeros(data_shape, dtype=cp.complex64)
    kernel = cp.ones(data_shape, dtype=cp.complex64)

    y = cp.fft.fft2(x)
    y_ = cp.fft.ifft2(y, norm="forward")

    assert y.shape == x.shape


def test_backward_double_callback_cpfft(callbacks):

    data_shape, backwardCallback = callbacks

    x = cp.zeros(data_shape, dtype=cp.complex64)
    kernel = cp.ones(data_shape, dtype=cp.complex64)

    with cp.fft.config.set_cufft_callbacks(
        cb_store=backwardCallback, cb_store_aux_arr=kernel
    ):
        y = cp.fft.fft2(x)
        # cp.fft.fft2(cp.from_dlpack(x))

    y_ = cp.fft.ifft2(y, norm="forward")
    # cp.fft.ifft2(y, norm = "forward")
    assert y.shape == x.shape


def test_backward_double_callback_cufft_planahead(callbacks):

    data_shape, backwardCallback = callbacks

    x = cp.zeros(data_shape, dtype=cp.complex64)
    kernel = cp.ones(data_shape, dtype=cp.complex64)
    plan = cufft.get_fft_plan(x, shape=data_shape, value_type="C2C")

    with cp.fft.config.set_cufft_callbacks(
        cb_store=backwardCallback, cb_store_aux_arr=kernel
    ):
        y = cufft.fft2(x, plan=plan)
        # cp.fft.fft2(cp.from_dlpack(x))

    y = cufft.ifft2(y, norm="forward", plan=plan)
    # cp.fft.ifft2(y, norm = "forward")
    assert y.shape == x.shape
