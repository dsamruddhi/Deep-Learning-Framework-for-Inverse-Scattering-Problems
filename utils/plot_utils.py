import io
import random
import matplotlib.pyplot as plt

from config import Config


class PlotUtils:

    @staticmethod
    def get_doi_extent():
        doi_length = Config.config["plot"]["doi_length"]
        doi_width = doi_length
        extent = [-doi_length/2, doi_length/2, -doi_width/2, doi_width/2]
        return extent

    @staticmethod
    def get_cmap():
        cmap = Config.config["plot"]["cmap"]
        if cmap == "jet":
            return plt.cm.jet

    @staticmethod
    def plot_results(test_output, test_input, pred_output):

        plot_extent = PlotUtils.get_doi_extent()
        plot_cmap = PlotUtils.get_cmap()

        for i in random.sample(range(0, test_input.shape[0]), 10):
            print(i)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)
            # fig.tight_layout()

            original = ax1.imshow(test_output[i, :, :], cmap=plot_cmap, extent=plot_extent)
            cb1 = fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
            cb1.ax.tick_params(labelsize=12)
            ax1.title.set_text(f"Original scatterer {i}")
            ax1.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

            guess_real = ax2.imshow(test_input[i, :, :, 0], cmap=plot_cmap, extent=plot_extent)
            cb2 = fig.colorbar(guess_real, ax=ax2, fraction=0.046, pad=0.04)
            cb2.ax.tick_params(labelsize=12)
            ax2.title.set_text("Initial guess: real")
            ax2.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

            guess_imag = ax3.imshow(test_input[i, :, :, 1], cmap=plot_cmap, extent=plot_extent)
            cb3 = fig.colorbar(guess_imag, ax=ax3, fraction=0.046, pad=0.04)
            cb3.ax.tick_params(labelsize=12)
            ax3.title.set_text("Initial guess: imaginary")
            ax3.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

            reconstruction = ax4.imshow(pred_output[i, :, :], cmap=plot_cmap, extent=plot_extent)
            cb4 = fig.colorbar(reconstruction, ax=ax4, fraction=0.046, pad=0.04)
            cb4.ax.tick_params(labelsize=12)
            ax4.title.set_text("Reconstructed")
            ax4.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

            plt.setp(ax1.get_xticklabels(), fontsize=12, horizontalalignment="left")
            plt.setp(ax2.get_xticklabels(), fontsize=12, horizontalalignment="left")
            plt.setp(ax3.get_xticklabels(), fontsize=12, horizontalalignment="left")
            plt.setp(ax4.get_xticklabels(), fontsize=12, horizontalalignment="left")

            plt.setp(ax1.get_yticklabels(), fontsize=12)
            plt.setp(ax2.get_yticklabels(), fontsize=12)
            plt.setp(ax3.get_yticklabels(), fontsize=12)
            plt.setp(ax4.get_yticklabels(), fontsize=12)

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
            plt.show()

    @staticmethod
    def plot_errors(test_output, pred_output, title, show):

        plot_extent = PlotUtils.get_doi_extent()
        plot_cmap = PlotUtils.get_cmap()

        fig, (ax1, ax2) = plt.subplots(ncols=2)

        original = ax1.imshow(test_output, cmap=plot_cmap, extent=plot_extent)
        cb1 = fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
        cb1.ax.tick_params(labelsize=12)
        ax1.title.set_text(f"Expected output")
        ax1.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        guess_real = ax2.imshow(pred_output, cmap=plot_cmap, extent=plot_extent)
        cb2 = fig.colorbar(guess_real, ax=ax2, fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=12)
        ax2.title.set_text("Obtained output")
        ax2.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        plt.setp(ax1.get_xticklabels(), fontsize=12, horizontalalignment="left")
        plt.setp(ax2.get_xticklabels(), fontsize=12, horizontalalignment="left")

        plt.setp(ax1.get_yticklabels(), fontsize=12)
        plt.setp(ax2.get_yticklabels(), fontsize=12)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

        if show:
            fig.suptitle(f"{title}")
            plt.show()

        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            return buf
