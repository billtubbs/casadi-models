import matplotlib.pyplot as plt


def make_io_plot(
    data,
    input_names,
    output_names,
    state_names=None,
    y_pred=None,
    time_label=None,
    plot_info=None,
):

    # data must be a Pandas dataframe with time as the index
    inputs = data[input_names]
    outputs = data[output_names]
    ny = outputs.shape[1]
    nu = inputs.shape[1]
    if time_label is None:
        time_label = data.index.name
    n_plots = ny + nu

    fig, axes = plt.subplots(
        n_plots, 1, figsize=(7, 1 + 1.5 * n_plots), sharex=True
    )

    for ax, name in zip(axes[:ny], output_names):
        outputs[name].plot(style=".", ax=ax)
        if y_pred is not None:
            y_pred.plot(label="pred", ax=ax)
            ax.legend()
        ax.grid(True)
        ax.set_title(f"Output - {name}")

    for ax, name in zip(axes[ny : ny + nu], input_names):
        inputs[name].plot(ax=ax, drawstyle="steps-post")
        ax.grid(True)
        ax.set_title(f"Input - {name}")

    return fig, axes
