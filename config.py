# Global variables
PATH_TO_DATA = "../data"
PATH_TO_FIGURES = "../figures"
PATH_TO_SOURCE = "../src"
PATH_TO_NOTEBOOKS = "../notebooks"

# Plotting
import seaborn as sns


def configure_plots():
    custom_params = {"axes.spines.right": False, "axes.spines.top": False, "figure.figsize": (9, 9),
                     "axes.labelcolor": "#696969", "axes.edgecolor": "#A9A9A9", "ytick.color": "#A9A9A9",
                     "xtick.color": "#A9A9A9", "legend.labelcolor": "#696969"}
    sns.set_theme(style="ticks", rc=custom_params, palette="colorblind", font_scale=1.2)



