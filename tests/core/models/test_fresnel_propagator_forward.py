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
    forwardCallback = r"""
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
                kernelEl.y *= 1./DATASIZE;
                
                cufftComplex value = ComplexMul(element, kernelEl);
                ((cufftComplex*)dataOut)[offset] = value;
                }}
                 __device__ cufftCallbackStoreC d_storeCallbackPtr = FresnelCallback;      
                """.format(
        arrsize=data_shape[0] * data_shape[1]
    )

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

    parametrizedCallback = r"""
                #include<inttypes.h>
                #include <cooperative_groups.h>
                
                __device__ __host__ inline 
                cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
                {   
                cufftComplex c;
                c.x = a.x * b.x - a.y * b.y;
                c.y = a.x * b.y + a.y * b.x;
                return c;
                }
                
__device__  void FresnelCallback(
                                void *dataOut,
                                size_t offset,
                                cufftComplex element,
                                void *callerInfo,
                                void *sharedPtr){

uint64_t *interface = (uint64_t*)callerInfo;
//cufftComplex *kernel = (cufftComplex*) interface[0];
uint64_t numel = interface[0];
uint64_t height = interface[1];
uint64_t width = interface[2];
double *Fr = (double*)&interface[3];
uint64_t direction = interface[4];
               
if(offset == 0) 
{   
    //printf("0");
    /*printf("Fresnelnumber: %f, Direction: %i, Fr_addr %" PRIu64 "\n", Fr[0], 
    direction, interface[3]);
    printf("width: %" PRIu64 ", height: %" PRIu64 ", numel: %" PRIu64 "\n", width, height, numel); 
    printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", interface[0],interface[1],interface[2],
    interface[3],
    interface[4] );*/
}
 //__syncthreads();

// compute kernel on the fly

size_t x_ = offset % width;
size_t y_ = offset / width;
long x = (x_ + (width/2))%width  - (width/2);
long y = (y_ + (height/2))%height  - (height/2);

float kx2 = 1.*x / float(width);
kx2 *= kx2;
float ky2 = 1.*y / float(height);
ky2 *= ky2;
float v = (-M_PI / (Fr[0]))*(kx2 + ky2);
float cosv;
float sinv;
sincosf(v, &sinv, &cosv);

cufftComplex kernelEl;
kernelEl.x = cosv;
kernelEl.y = sinv;
//kernelEl = kernel[offset];

kernelEl.x *= 1./numel;
if (direction == 1) {
    kernelEl.y *= 1./numel;
}
else //backward case
{
    kernelEl.y *= -1./numel;
}

//printf("i am working: x=%d, y=%d, x_=%d, y_=%d, val=(%f,%f)\n",
// x ,y, x_, y_, cosv, sinv);


cufftComplex value = ComplexMul(element, kernelEl);

((cufftComplex*)dataOut)[offset] = value;

}
 __device__ cufftCallbackStoreC d_storeCallbackPtr = FresnelCallback;      
                """

    yield data_shape, forwardCallback, backwardCallback


def test_forward_callback(callbacks):

    data_shape, forwardCallback, _ = callbacks

    x = cp.zeros(data_shape, dtype=cp.complex64)
    kernel = cp.ones(data_shape, dtype=cp.complex64)
    with cp.fft.config.set_cufft_callbacks(
        cb_store=forwardCallback, cb_store_aux_arr=kernel
    ):
        y = cp.fft.fft2(cp.from_dlpack(x))

    assert y.shape == x.shape


def test_forward_no_callback_plain(callbacks):

    data_shape, _, _ = callbacks

    x = cp.zeros(data_shape, dtype=cp.complex64)
    kernel = cp.ones(data_shape, dtype=cp.complex64)

    # with pytest.raises(Exception):
    y = cp.fft.fft2(x) * kernel

    assert y.shape == x.shape


def test_forward_no_callback_cufft(callbacks):

    data_shape, _, _ = callbacks

    x = cp.zeros(data_shape, dtype=cp.complex64)
    kernel = cp.ones(data_shape, dtype=cp.complex64)

    y = cufft.fft2(x) * kernel

    assert y.shape == x.shape


def test_forward_no_callback_cufft_planahead(callbacks):

    data_shape, _, _ = callbacks

    x = cp.zeros(data_shape, dtype=cp.complex64)
    kernel = cp.ones(data_shape, dtype=cp.complex64)

    plan = cufft.get_fft_plan(x, shape=data_shape, value_type="C2C")

    y = cufft.fft2(x, plan=plan) * kernel

    assert y.shape == x.shape


def test_forward_roundtrips_callback(callbacks):

    data_shape, forwardCallback, _ = callbacks

    x = cp.zeros(data_shape, dtype=cp.complex64)
    kernel = cp.ones(data_shape, dtype=cp.complex64)

    exp = cp.reshape(cp.random.randn(x.size, dtype=cp.float32), data_shape)
    y = cp.zeros(data_shape, dtype=cp.complex64)
    y.real = cp.copy(exp)
    y.imag = cp.copy(exp)

    with cp.fft.config.set_cufft_callbacks(
        cb_store=forwardCallback, cb_store_aux_arr=kernel
    ):
        # y = cp.fft.fft2(cp.from_dlpack(x))
        y = cp.fft.fft2(x)

    y = cp.fft.ifft2(y, norm="forward")

    assert hasattr(y, "shape")
    assert y.shape == x.shape
