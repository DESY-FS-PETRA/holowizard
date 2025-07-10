from holowizard.core.utils.fileio import load_motor_log


class P05_Scan:
    def __init__(self, scanname, energy, holder, path_raw, qp=True):
        self.scanname = scanname
        self.path_raw = path_raw

        self.energy = energy
        self.holder = holder
        self.fzp_dr = 50  # nm
        self.fzp_d = 300  # um
        self.wl = 1.2398 / self.energy
        self.qp = qp
        self.calc_geometry_p05()

    def calc_geometry_p05(self):
        motor_pos = load_motor_log(self.path_raw)
        o_stage_y = float(motor_pos["OpticsStage1_y"])
        slider1 = float(motor_pos["GraniteSlab_1"])
        slider2 = float(motor_pos["GraniteSlab_2"])
        sf1_y = float(motor_pos["OpticsSF1_y"])

        if self.qp:
            detDistInit = 20413  # mm
            posInit = -80.77  #
        else:  # distances for old goldi
            detDistInit = 20264  # mm
            posInit = -66  #

        corr = 0
        offset = slider1 + o_stage_y + sf1_y + self.holder + self.fzp_f  # % mm

        self.z02 = (detDistInit - offset) * 1e6
        self.z01 = (posInit + slider2 - offset + corr) * 1e6
        return self.z01, self.z02

    # calculates focal distance of FZP, in mm
    @property
    def fzp_f(self):
        return self.fzp_d * 1e-3 * self.fzp_dr * 1e-6 / (self.wl * 1e-6)

    # @property
    # def z01(self):
    #     return self.z01
    #
    # @property
    # def z02(self):
    #     return self.z02
    #
    # @z01.setter
    # def z01(self,value):
    #     self.z01 = value
    #
    # @z02.setter
    # def z02(self, value):
    #     self.z02 = value
