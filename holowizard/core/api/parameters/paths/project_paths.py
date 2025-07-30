from holowizard.core.parameters.paths import Paths


class ProjectPaths(Paths):
    def __init__(
        self,
        session_name,
        session_id,
        output_dir="",
        flatfield_components_name="flatfield_components.pkl",
        root_dir=None,
        other: Paths = None,
        **kwargs,
    ):
        super().__init__(root_dir, other)

        if other is not None:
            self.__dict__.update(other.get_dict())

        self.session_name = str(session_name)
        self.session_id = str(session_id)
        self.temp_dir = self.root_dir + "/temp/"
        self.output_dir = output_dir
        if "flatfield_components_file" in kwargs.keys():
            self.flatfield_components_file = kwargs["flatfield_components_file"]
        else:
            self.flatfield_components_file = self.temp_dir + flatfield_components_name
        self.params_dir = self.root_dir + "/params/"
        self.logs_dir = self.root_dir + "/logs/"
        self.session_logs_name = self.session_name + "_" + str(self.session_id)
        self.results_dir = self.temp_dir
        self.result_phaseshift_file = self.results_dir + "image_phaseshift_" + str(self.session_id) + ".tiff"
        self.result_absorption_file = self.results_dir + "image_absorption_" + str(self.session_id) + ".tiff"
        self.io_file = self.temp_dir + "image" + str(self.session_id) + ".tiff"
        self.se_loss_records_file = self.logs_dir + "se_losses.csv"

        self.__dict__.update(kwargs)
        self.update()

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def temp_dir(self):
        return self.dirs["temp_dir"]

    @property
    def output_dir(self):
        return self.dirs["output_dir"]

    @property
    def session_name(self):
        return self.names["session_name"]

    @property
    def session_id(self):
        return self.names["session_id"]

    @property
    def flatfield_components_file(self):
        return self.files["flatfield_components_file"]

    @property
    def params_dir(self):
        return self.dirs["params_dir"]

    @property
    def logs_dir(self):
        return self.dirs["logs_dir"]

    @property
    def session_logs_name(self):
        return self.names["session_logs_name"]

    @property
    def results_dir(self):
        return self.dirs["results_dir"]

    @property
    def result_phaseshift_file(self):
        return self.files["result_phaseshift_file"]

    @property
    def result_absorption_file(self):
        return self.files["result_absorption_file"]

    @property
    def io_file(self):
        return self.files["io_file"]

    @property
    def se_loss_records_file(self):
        return self.files["se_loss_records_file"]

    @temp_dir.setter
    def temp_dir(self, temp_dir):
        self.dirs["temp_dir"] = temp_dir

    @output_dir.setter
    def output_dir(self, output_dir):
        self.dirs["output_dir"] = output_dir

    @session_name.setter
    def session_name(self, session_name):
        self.names["session_name"] = session_name

        try:
            self.session_logs_name = session_name + "_" + str(self.session_id)
        except:
            pass

    @session_id.setter
    def session_id(self, session_id):
        self.names["session_id"] = session_id

        try:
            self.session_logs_name = self.session_name + "_" + str(session_id)
        except:
            pass

    @flatfield_components_file.setter
    def flatfield_components_file(self, flatfield_components_file):
        self.files["flatfield_components_file"] = flatfield_components_file

    @params_dir.setter
    def params_dir(self, params_dir):
        self.dirs["params_dir"] = params_dir

    @logs_dir.setter
    def logs_dir(self, logs_dir):
        self.dirs["logs_dir"] = logs_dir

    @session_logs_name.setter
    def session_logs_name(self, session_logs_name):
        self.names["session_logs_name"] = session_logs_name

    @results_dir.setter
    def results_dir(self, results_dir):
        self.dirs["results_dir"] = results_dir

    @result_phaseshift_file.setter
    def result_phaseshift_file(self, result_phaseshift_file):
        self.files["result_phaseshift_file"] = result_phaseshift_file

    @result_absorption_file.setter
    def result_absorption_file(self, result_absorption_file):
        self.files["result_absorption_file"] = result_absorption_file

    @io_file.setter
    def io_file(self, io_file):
        self.files["io_file"] = io_file

    @se_loss_records_file.setter
    def se_loss_records_file(self, se_loss_records_file):
        self.files["se_loss_records_file"] = se_loss_records_file
