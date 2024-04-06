from corner import corner as cornerplot


def generate_cornerplot(
    samples,
    weights,
    group_name,
    parameter_names,
    parameter_range,
    confidence=0.99,
    existing_figure=None,
    existing_titles=None,
    existing_title_color=None,
    color="#53C79B",
    MLE_color="gold",
    MLE_values=None,
    MLE_name="MLE value",
    overtext_y=0.5,
    string_formats=".4g",
    reference_values=None,
    reference_lowerbound=None,
    reference_upperbound=None,
    reference_name="Reference",
    reference_markerstyle="*",
    reference_color="gold",
    plot_generic_legend_labels=False,
):
    if existing_figure is None:
        fig = cornerplot(
            samples,
            weights=weights,
            bins=20,
            color=color,
            labels=parameter_names,
            quantiles=[0.16, 0.84],
            range=parameter_range,
            # smooth=0.3,
            # smooth1d=0.3,
            hist_kwargs={"facecolor": color, "alpha": 0.5},
            show_titles=True,
            title_fmt=string_formats[0],
            title_kwargs={"pad": 0, "color": color},
            title_quantiles=[0.16, 0.84],
            plot_datapoints=True,
            labelsize=11,
            use_math_text=True,
        )
        fig.subplots_adjust(left=0.10, bottom=0.10, wspace=0.075, hspace=0.075)
    else:
        fig = cornerplot(
            samples,
            fig=existing_figure,
            weights=weights,
            bins=20,
            color=color,
            labels=parameter_names,
            quantiles=[0.16, 0.84],
            range=parameter_range,
            # smooth=0.3,
            # smooth1d=0.3,
            hist_kwargs={"facecolor": color, "alpha": 0.5},
            show_titles=True,
            title_fmt=string_formats[0],
            title_kwargs={"pad": 10, "color": color},
            title_quantiles=[0.16, 0.84],
            plot_datapoints=True,
            labelsize=11,
            use_math_text=True,
        )
        fig.subplots_adjust(left=0.10, bottom=0.10, wspace=0.075, hspace=0.075)

    par_titles = []

    if reference_values is None:
        reference_values = np.full_like(MLE_values, np.nan)

    # Extract the axes
    ndim = len(parameter_names)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # if MLE_values is not None:
    #    retrieved_values = MLE_values
    # else:
    #    retrieved_values = np.nanpercentile(samples,50,axis=0)
    retrieved_values = np.nanpercentile(samples, 50, axis=0)

    cumulative_xlims = []
    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        xlim = ax.xaxis.get_data_interval()
        if not np.isnan(reference_values[i]):
            xlim[0] = np.min(
                [xlim[0], reference_values[i] - 0.1 * np.abs(np.ptp(xlim))]
            )
            xlim[1] = np.max(
                [xlim[1], reference_values[i] + 0.1 * np.abs(np.ptp(xlim))]
            )
            formatted_syntax = r"{:" + string_formats[i] + r"}"
            reference_overtext = ax.text(
                reference_values[i],
                np.sum(ax.yaxis.get_data_interval()) / 5,
                formatted_syntax.format(reference_values[i]),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=18,
                color=reference_color,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="#444444")],
            )
            if reference_lowerbound is not None:
                ax.axvline(
                    reference_lowerbound[i],
                    linestyle="dashed",
                    linewidth=2,
                    color=reference_color,
                )
                if reference_lowerbound[i] < xlim[0]:
                    xlim[0] = reference_lowerbound[i]
            if reference_upperbound is not None:
                ax.axvline(
                    reference_upperbound[i],
                    linestyle="dashed",
                    linewidth=2,
                    color=reference_color,
                )
                if reference_upperbound[i] > xlim[1]:
                    xlim[1] = reference_upperbound[i]

        formatted_syntax = r"{:" + string_formats[i] + r"}"
        reference_overtext = ax.text(
            MLE_values[i],
            overtext_y * np.sum(ax.yaxis.get_data_interval()),
            formatted_syntax.format(MLE_values[i]),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=18,
            color=MLE_color,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="#444444")],
        )

        ax.set_xlim(xlim)
        cumulative_xlims.append(xlim)
        ylim = ax.yaxis.get_data_interval()
        ax.set_ylim(ylim)
        ax.margins(0.05)

        if group_name == "clouds":
            hist_title_fontsize = 12
        else:
            hist_title_fontsize = 16

        # USE CONDITIONAL FOR OVERPLOTTING
        if existing_figure is None:
            par_title = ax.get_title()
            ax.set_title(
                par_title,
                fontsize=hist_title_fontsize,
                color=color,
                path_effects=[pe.withStroke(linewidth=1, foreground="#444444")],
            )
            par_titles.append(par_title)

        else:
            par_overtitle = ax.get_title()
            ax.text(
                0.5,
                1.25,
                par_overtitle,
                fontsize=hist_title_fontsize,
                color=color,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                path_effects=[pe.withStroke(linewidth=1, foreground="#444444")],
            )
            if existing_titles is not None:
                ax.set_title(
                    existing_titles[i],
                    fontsize=hist_title_fontsize,
                    color=existing_title_color,
                    path_effects=[pe.withStroke(linewidth=1, foreground="#444444")],
                )

        title_color = color

        # ax.axvline(retrieved_values[i], linewidth=3, color="#444444")
        # ax.axvline(retrieved_values[i], color=color)

        ax.axvline(MLE_values[i], linewidth=3, color="#444444")
        ax.axvline(MLE_values[i], color=MLE_color)

        if not np.isnan(reference_values[i]):
            ax.axvline(reference_values[i], linewidth=3, color="#444444")
            ax.axvline(reference_values[i], color=reference_color)

    # Loop over the histograms
    for yi in range(ndim):
        for xi, xlim in enumerate(cumulative_xlims[:yi]):
            ax = axes[yi, xi]
            ax.set_xlim(xlim)
            ylim = ax.yaxis.get_data_interval()
            if not np.isnan(reference_values[yi]):
                ylim[0] = np.min(
                    [ylim[0], reference_values[yi] - 0.1 * np.abs(np.ptp(ylim))]
                )
                ylim[1] = np.max(
                    [ylim[1], reference_values[yi] + 0.1 * np.abs(np.ptp(ylim))]
                )
            ax.set_ylim(ylim)
            ax.margins(0.05)

            # ax.axvline(retrieved_values[xi], linewidth=3, color="#444444")
            # ax.axhline(retrieved_values[yi], linewidth=3, color="#444444")
            # ax.axvline(retrieved_values[xi], color=color)
            # ax.axhline(retrieved_values[yi], color=color)

            ax.axvline(MLE_values[xi], linewidth=3, color="#444444")
            ax.axhline(MLE_values[yi], linewidth=3, color="#444444")
            ax.axvline(MLE_values[xi], color=MLE_color)
            # MLE_locator = ax.axhline(MLE_values[yi], color=MLE_color, label=MLE_name)
            MLE_locator = ax.axhline(MLE_values[yi], color=MLE_color)

            if not np.isnan(reference_values[yi]) and not np.isnan(
                reference_values[xi]
            ):
                ax.axvline(reference_values[xi], linewidth=3, color="#444444")
                ax.axvline(reference_values[xi], color=reference_color)
                ax.axhline(reference_values[yi], linewidth=3, color="#444444")
                ax.axhline(reference_values[yi], color=reference_color)

            # ax.plot(retrieved_values[xi], retrieved_values[yi], marker="o", markersize=10, color="#444444")
            # median_marker, = ax.plot(retrieved_values[xi], retrieved_values[yi], marker="o", markersize=8, color=color)
            ax.plot(
                MLE_values[xi],
                MLE_values[yi],
                marker="D",
                markersize=10,
                color="#444444",
            )
            (MLE_marker,) = ax.plot(
                MLE_values[xi],
                MLE_values[yi],
                marker="D",
                markersize=8,
                color=MLE_color,
            )

            if not np.isnan(reference_values[yi]) and not np.isnan(
                reference_values[xi]
            ):
                if xi == 0 and yi == 1 and reference_name is not None:
                    (reference_locator,) = ax.plot(
                        reference_values[xi],
                        reference_values[yi],
                        marker=reference_markerstyle,
                        markersize=10,
                        color="#444444",
                    )
                    (reference_marker,) = ax.plot(
                        reference_values[xi],
                        reference_values[yi],
                        marker=reference_markerstyle,
                        markersize=8,
                        color=reference_color,
                    )
                    print("We're setting the reference label!")
                else:
                    ax.plot(
                        reference_values[xi],
                        reference_values[yi],
                        marker=reference_markerstyle,
                        markersize=10,
                        color="#444444",
                    )
                    ax.plot(
                        reference_values[xi],
                        reference_values[yi],
                        marker=reference_markerstyle,
                        markersize=8,
                        color=reference_color,
                    )

    if reference_name is not None:
        reference_marker.set_label("{} value".format(reference_name))
    # MLE_locator.set_label(MLE_name)

    if plot_generic_legend_labels:
        legend_handles, legend_labels = axes[1, 0].get_legend_handles_labels()
        legend_handles.append(
            Line2D([0], [0], marker="D", markersize=10, color="#444444")
        )
        # legend_handles.append(Line2D([0], [0], marker="D", markersize=10, color=MLE_color))
        legend_labels.append("MLE value")
        legend_handles.append(
            Line2D([0], [0], linestyle="dashed", linewidth=3, color="#444444")
        )
        # legend_handles.append(Line2D([0], [0], linestyle="dashed", linewidth=3, color=MLE_color))
        legend_labels.append("68\% confidence interval")

    # median_marker.set_label("Median value")
    # USE CONDITIONAL FOR OVERPLOTTING
    if plot_generic_legend_labels:
        plt.figlegend(
            handles=legend_handles,
            labels=legend_labels,
            fontsize=28,
            loc="upper right",
            facecolor="#444444",
            framealpha=0.25,
        )
    return fig, par_titles, title_color
