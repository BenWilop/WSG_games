# In a new Jupyter cell

import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
from wsg_games.tictactoe.crosscoder.crosscoder_metrics import CrosscoderMetrics


class CrosscoderFeatureExplorer:
    def __init__(self, metrics: CrosscoderMetrics):
        self.metrics = metrics
        self.output = widgets.Output()

        # Pre-calculate plottable indices to skip empty features
        self.plottable_indices = {
            j for j, games in self.metrics.top_n_activations.items() if games
        }

        self.index_sets = {
            "All": sorted(list(self.plottable_indices)),
            "Model 1": sorted(
                list(set(self.metrics.model_1_features) & self.plottable_indices)
            ),
            "Shared": sorted(
                list(set(self.metrics.shared_features) & self.plottable_indices)
            ),
            "Model 2": sorted(
                list(set(self.metrics.model_2_features) & self.plottable_indices)
            ),
        }

        self.active_set_name = "All"
        self.active_indices = self.index_sets[self.active_set_name]

        self._create_widgets()
        self._link_widgets()

    def _create_widgets(self):
        max_feat = self.metrics.crosscoder.dict_size - 1
        self.slider = widgets.IntSlider(
            min=0, max=max_feat, step=1, description="Feature Index:"
        )
        self.player = widgets.Play(
            min=0, max=max_feat, step=1, interval=1000, show_repeat=False
        )
        self.text = widgets.IntText(value=0, description="Jump to:")

        self.filter_buttons = widgets.ToggleButtons(
            options=[
                (f"All ({len(self.index_sets['All'])})", "All"),
                (f"Model 1 ({len(self.index_sets['Model 1'])})", "Model 1"),
                (f"Shared ({len(self.index_sets['Shared'])})", "Shared"),
                (f"Model 2 ({len(self.index_sets['Model 2'])})", "Model 2"),
            ],
            value="All",
            description="Filter Features:",
            style={"button_width": "120px"},
        )

        self.prev_button = widgets.Button(
            description="◀ Prev", layout={"width": "80px"}
        )
        self.next_button = widgets.Button(
            description="Next ▶", layout={"width": "80px"}
        )

    def _link_widgets(self):
        # Link slider and text box together
        widgets.jslink((self.slider, "value"), (self.text, "value"))
        widgets.jslink((self.player, "value"), (self.text, "value"))

        # Define actions
        self.slider.observe(self._on_value_change, names="value")
        self.filter_buttons.observe(self._on_filter_change, names="value")
        self.prev_button.on_click(self._on_prev_click)
        self.next_button.on_click(self._on_next_click)

    def _on_value_change(self, change):
        self._update_plot()

    def _on_filter_change(self, change):
        self.active_set_name = change.new
        self.active_indices = self.index_sets[self.active_set_name]
        # Jump to the first available feature in the new set
        if self.active_indices:
            self.slider.value = self.active_indices[0]
        self._update_plot()

    def _on_prev_click(self, b):
        current_val = self.slider.value
        # Find the index of the first element smaller than the current value
        search_result = np.searchsorted(self.active_indices, current_val)
        if search_result > 0:
            self.slider.value = self.active_indices[search_result - 1]
        else:  # Wrap around to the end
            self.slider.value = self.active_indices[-1]

    def _on_next_click(self, b):
        current_val = self.slider.value
        # Find the index of the first element larger than the current value
        search_result = np.searchsorted(self.active_indices, current_val, side="right")
        if search_result < len(self.active_indices):
            self.slider.value = self.active_indices[search_result]
        else:  # Wrap around to the start
            self.slider.value = self.active_indices[0]

    def _update_plot(self):
        with self.output:
            clear_output(wait=True)
            feature_idx = self.slider.value
            if feature_idx in self.plottable_indices:
                fig = self.metrics.display_feature_information(feature_idx)
                plt.show(fig)
            else:
                print(
                    f"Feature {feature_idx} has no top-activating games and is skipped."
                )

    def display(self):
        # Initial plot
        self._update_plot()

        # Arrange and display widgets
        top_controls = widgets.HBox([self.slider, self.text, self.player])
        nav_controls = widgets.HBox(
            [self.filter_buttons, self.prev_button, self.next_button]
        )
        display(widgets.VBox([top_controls, nav_controls, self.output]))
