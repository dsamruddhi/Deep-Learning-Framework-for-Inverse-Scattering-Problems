import matplotlib.pyplot as plt

import config


class PlotUtils:

    @staticmethod
    def get_doi_extent():
        doi_length = config.config["plot"]["doi_length"]
        doi_width = doi_length
        extent = [-doi_length/2, doi_length/2, -doi_width/2, doi_width/2]
        return extent

    @staticmethod
    def get_cmap():
        cmap = config.config["plot"]["cmap"]
        if cmap == "jet":
            return plt.cm.jet
