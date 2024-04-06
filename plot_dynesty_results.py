# %%
from TP_profiles import Piette

from visualization_functions import create_linear_colormap

# %%
cmap_kwargs = {
    "lightness_minimum": 0.15,
    "lightness_maximum": 0.85,
    "saturation_minimum": 0.2,
    "saturation_maximum": 0.8,
}

cmap_H2O = create_linear_colormap(["#226666", "#2E4172"], **cmap_kwargs)
cmap_CO = create_linear_colormap(["#882D60", "#AA3939"], **cmap_kwargs)
cmap_CO2 = create_linear_colormap(["#96A537", "#669933"], **cmap_kwargs)
cmap_CH4 = create_linear_colormap(["#96A537", "#669933"], **cmap_kwargs)

cmap_cloudy = create_linear_colormap(
    [cnames["lightcoral"], cnames["lightcoral"]], **cmap_kwargs
)
cmap_clear = create_linear_colormap(
    [cnames["cornflowerblue"], cnames["cornflowerblue"]], **cmap_kwargs
)

cmap_cloud = plt.get_cmap("Greys")

plotted_components = ["h2o", "co", "co2", "ch4"]
plotted_titles = ["H$_2$O", "CO", "CO$_2$", "CH$_4$"]
cmaps = [cmap_H2O, cmap_CO, cmap_CO2, cmap_CH4]

# %%
contributions_max = np.log10(
    np.nanmax(
        [
            np.nanmax(contribution)
            for (species, contribution) in JHK_contributions[directory].items()
            if species not in ["gas", "cloud", "total"]
        ]
    )
)
contributions_min = contributions_max - 3

wavelengths_low, wavelengths_high, data, data_error = (
    np.genfromtxt(DIRECTORY_RESULTS_2M2236 / OBSERVED_SPECTRUM_FILE_2M2236).T
)[:4]

band_breaks = np.r_[
    0,
    np.nonzero(
        (
            JHK_contributions[directory]["h2o"].index.to_numpy()[1:]
            - JHK_contributions[directory]["h2o"].index.to_numpy()[:-1]
        )
        > 0.05
    )[0]
    + 1,
    len(JHK_contributions[directory]["h2o"].index),
]
break_indices = np.where(wavelengths_low[1:] != wavelengths_high[:-1])[0]
mask = break_indices

padding = 0.025
number_of_bands = 2
wavelength_ranges = np.array(
    [
        JHK_contributions[directory]["h2o"].index[band_breaks[i + 1] - 1]
        - JHK_contributions[directory]["h2o"].index[band_breaks[i]]
        for i in range(number_of_bands)
    ]
)

fig = plt.figure(figsize=(40, 30))
gs = fig.add_gridspec(
    nrows=6,
    ncols=2,
    height_ratios=[4, 2, 3, 3, 3, 3],
    width_ratios=(1 + 2 * padding) * wavelength_ranges,
    wspace=0.1,
)

spectrum_lines = []
spectrum_axes = []
residual_axes = []
contribution_columns = []
for j, (directory, object_header, data_file, color, model_title) in enumerate(
    zip(
        directories,
        object_headers,
        data_files,
        plotting_colors,
        ["Cloudy model", "Clear model"],
    )
):
    spectrum_dict = extract_spectra(
        directory,
        object_header,
        DIRECTORY_RESULTS_2M2236 / MLE_SPECTRUM_FILE_2M2236,
        DIRECTORY_RESULTS_2M2236 / OBSERVED_SPECTRUM_FILE_2M2236,
    )

    datas = spectrum_dict["data"]
    models = spectrum_dict["MLE spectrum"]
    errors = spectrum_dict["data errors"]
    residuals = (models - datas) / errors

    # ymin = np.min([np.min(data), np.min(model)])
    # ymax = np.max([np.max(data), np.max(model)])
    # ymin = ymin - padding*np.abs(ymax-ymin)
    # ymax = ymax + padding*np.abs(ymax-ymin)

    residual_ymin = np.nanmin(residuals)
    residual_ymax = np.nanmax(residuals)
    residual_ymin = residual_ymin - padding * np.abs(residual_ymax - residual_ymin)
    residual_ymax = residual_ymax + padding * np.abs(residual_ymax - residual_ymin)

    concocted_boundaries = [[2.85, 4.01], [4.19, 5.30]]
    for i, band_boundaries in enumerate(concocted_boundaries):
        if j == 0:
            if i > 0:
                spectrum_ax = fig.add_subplot(gs[0, i])  # sharey=spectrum_axes[i-1])
                spectrum_axes.append(spectrum_ax)

                residual_ax = fig.add_subplot(
                    gs[1, i], sharex=spectrum_ax
                )  # sharey=residual_axes[i-1])
                residual_axes.append(residual_ax)
            else:
                spectrum_ax = fig.add_subplot(gs[0, i])
                spectrum_axes.append(spectrum_ax)

                residual_ax = fig.add_subplot(gs[1, i], sharex=spectrum_ax)
                residual_axes.append(residual_ax)
        else:
            spectrum_ax = spectrum_axes[i]
            residual_ax = residual_axes[i]

        # if i==len(concocted_boundaries)-1:
        #    spectrum_ax.text(
        #        0.975, 0.95-0.15*j,
        #        model_title,
        #        horizontalalignment="right",
        #        verticalalignment="top",
        #        transform=spectrum_ax.transAxes,
        #        fontsize=36,
        #        color=color
        #        )

        band_condition = (spectrum_dict["wavelengths"] > band_boundaries[0]) & (
            spectrum_dict["wavelengths"] < band_boundaries[1]
        )
        wavelengths = spectrum_dict["wavelengths"][band_condition]
        data = spectrum_dict["data"][band_condition]
        error = spectrum_dict["data errors"][band_condition]
        model = spectrum_dict["MLE spectrum"][band_condition]
        residual = residuals[band_condition]

        xmin = np.min(wavelengths)
        xmax = np.max(wavelengths)
        xmin = xmin - padding * np.abs(xmax - xmin)
        xmax = xmax + padding * np.abs(xmax - xmin)

        spectrum_ax.set_xlim([xmin, xmax])
        # spectrum_ax.set_ylim([ymin, ymax])

        linestyles = ["solid", "solid"]
        spectrum_ax.errorbar(
            wavelengths,
            data,
            error,
            color="#444444",
            fmt="x",
            linewidth=0,
            elinewidth=2,
            alpha=1,
            zorder=-3,
        )
        spectrum_ax.plot(
            wavelengths,
            model,
            color=color,
            linewidth=3,
            linestyle=linestyles[j],
            alpha=1,
            zorder=2 - j,
            label=model_title,
        )

        # if i==0 and j==0:
        # [spectrum_ax.axvline(line_position, linestyle="dashed", linewidth=1.5, zorder=-10, color="#888888")
        # for line_position in [1.139, 1.141, 1.169, 1.177, 1.244, 1.253, 1.268]]
        # y_text = spectrum_ax.get_ylim()[0] + 0.1*np.diff(spectrum_ax.get_ylim())
        # spectrum_ax.text((1.169+1.177)/2, y_text, "KI", fontsize=20, horizontalalignment="center")
        # spectrum_ax.text((1.244+1.253)/2, y_text, "KI", fontsize=20, horizontalalignment="center")
        # spectrum_ax.text(1.268, y_text, "NaI", fontsize=20, horizontalalignment="center")

        residual_ax.plot(
            wavelengths,
            residual,
            color=color,
            linewidth=3,
            linestyle=linestyles[j],
            alpha=1,
            zorder=2 - j,
        )
        residual_ax.axhline(
            0, color="#444444", linewidth=2, linestyle="dashed", zorder=-10
        )
        residual_ax.set_ylim([residual_ymin, residual_ymax])
        residual_ax.minorticks_on()

        if i == 0:
            spectrum_ax.set_ylabel("Flux (erg s$^{-1}$ cm$^{-3}$)", fontsize=36)
            residual_ax.set_ylabel(r"Residual/$\sigma$", fontsize=26)
        if (j == 1) and (i == len(concocted_boundaries) - 1):
            legend_handles, legend_labels = spectrum_ax.get_legend_handles_labels()
            legend_handles.append(Patch(facecolor="k", edgecolor="#DDDDDD"))
            legend_labels.append(r"Cloud $\tau > 0.1$")
            spectrum_ax.legend(
                handles=legend_handles, labels=legend_labels, fontsize=22, frameon=False
            )  # , loc="upper center")

        # residual = (model-data)/error
        reduced_chi_squared = np.sum(residual**2) / (np.shape(residual)[0] - npars)
        print("Reduced chi squared is {}".format(reduced_chi_squared))

        spectrum_ax.tick_params(axis="x", labelsize=26)
        spectrum_ax.tick_params(axis="y", labelsize=26)
        spectrum_ax.yaxis.get_offset_text().set_size(26)

        residual_ax.tick_params(axis="x", labelsize=26)
        residual_ax.tick_params(axis="y", labelsize=26)

        # CONTRIBUTION PLOTS

        contribution_axes = []
        for n, (comp, cmap, title) in enumerate(
            zip(plotted_components, cmaps, plotted_titles)
        ):
            if j == 0:
                contribution_ax = fig.add_subplot(gs[n + 2, i], sharex=spectrum_axes[i])
                contribution_axes.append(contribution_ax)
                if n == len(plotted_components) - 1:
                    contribution_columns.append(contribution_axes)
                if i == 0:
                    contribution_ax.text(
                        0.025,
                        0.05,
                        title + " contribution",
                        horizontalalignment="left",
                        verticalalignment="bottom",
                        transform=contribution_ax.transAxes,
                        fontsize=32,
                    )
            else:
                contribution_ax = contribution_columns[i][n]

            if j == 0:
                cont_cmap = cmap_cloudy
                outline_color = "lightcoral"
            else:
                cont_cmap = cmap_clear
                outline_color = "cornflowerblue"

            cf = JHK_contributions[directory][comp].to_numpy()
            x, y = np.meshgrid(
                JHK_contributions[directory][comp].index,
                JHK_contributions[directory][comp].columns,
            )
            contribution_ax.contourf(
                x[:, band_breaks[i] : band_breaks[i + 1] : 8],
                y[:, band_breaks[i] : band_breaks[i + 1] : 8],
                np.log10(cf).T[:, band_breaks[i] : band_breaks[i + 1] : 8],
                cmap=cont_cmap,
                levels=contributions_max - np.array([4, 2, 0]),
                alpha=0.66,
                zorder=0,
            )
            contribution_ax.contour(
                x[:, band_breaks[i] : band_breaks[i + 1] : 8],
                y[:, band_breaks[i] : band_breaks[i + 1] : 8],
                np.log10(cf).T[:, band_breaks[i] : band_breaks[i + 1] : 8],
                colors=outline_color,
                levels=contributions_max - np.array([2]),
                linewidths=3,
                alpha=1,
                zorder=1,
            )
            if j == 0:
                contribution_ax.invert_yaxis()
            if "cloud" in JHK_contributions[directory]:
                cloud_cf = JHK_contributions[directory]["cloud"]
                x, y = np.meshgrid(cloud_cf.index, cloud_cf.columns)
                contribution_ax.contourf(
                    x[:, band_breaks[i] : band_breaks[i + 1] : 8],
                    y[:, band_breaks[i] : band_breaks[i + 1] : 8],
                    cloud_cf.to_numpy().T[:, band_breaks[i] : band_breaks[i + 1] : 8],
                    colors="k",
                    # cmap=cmap_cloud,
                    alpha=0.75,
                    # levels=np.logspace(-1, 2, num=20),
                    levels=[0.1, 0.75],
                    zorder=2,
                )
                if i == 0:
                    contribution_ax.contour(
                        x[:, band_breaks[i] : band_breaks[i + 1] : 8],
                        y[:, band_breaks[i] : band_breaks[i + 1] : 8],
                        cloud_cf.to_numpy().T[
                            :, band_breaks[i] : band_breaks[i + 1] : 8
                        ],
                        colors="#DDDDDD",
                        linestyles="solid",
                        linewidth=3,
                        alpha=1,
                        levels=[0.1],
                        zorder=3,
                    )
                contribution_ax.tick_params(axis="x", labelsize=26)
                contribution_ax.tick_params(axis="y", labelsize=26)
                contribution_ax.minorticks_on()
            # contribution_ax.contour(x[:, band_breaks[i]:band_breaks[i+1]:8], y[:, band_breaks[i]:band_breaks[i+1]:8],
            #                       np.log10(cf).T[:, band_breaks[i]:band_breaks[i+1]:8],
            #                       cmap=cmap,
            #                       levels=contributions_max-np.array([3, 2, 1, 0]),
            #                       alpha=1,
            #                       zorder=0)

contributions_ax = fig.add_subplot(gs[2:, :])
contributions_ax.spines["top"].set_color("none")
contributions_ax.spines["bottom"].set_color("none")
contributions_ax.spines["left"].set_color("none")
contributions_ax.spines["right"].set_color("none")
contributions_ax.tick_params(
    labelcolor="none", top=False, bottom=False, left=False, right=False
)
contributions_ax.grid(False)
contributions_ax.set_xlabel(
    r"$\lambda\left(\mu\mathrm{m}\right)$", fontsize=36, y=-0.075
)
contributions_ax.set_ylabel(
    r"$\log_{10}\left(P/\mathrm{bar}\right)$", fontsize=36, labelpad=20
)

"""
overall_ax = fig.add_subplot(111)
overall_ax.spines['top'].set_color('none')
overall_ax.spines['bottom'].set_color('none')
overall_ax.spines['left'].set_color('none')
overall_ax.spines['right'].set_color('none')
overall_ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
overall_ax.grid(False)
overall_ax.set_xlabel(r"$\lambda\left(\mu\mathrm{m}\right)$", fontsize=36, y=-0.075)
"""


for filetype in filetypes:
    plt.savefig(
        object_label + ".fit-spectrum+contributions.{}".format(filetype),
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )

# %%
fig, ax = plt.subplots(figsize=(8, 6))
for (band, samples), color, label in zip(
    sample_dicts.items(), plotting_colors, ["cloudy", "clear"]
):
    T_samples = (samples["T-P"][2]).T
    MLE_Tsample = samples["T-P"][3]
    logP, [T_minus_1sigma, T_median, T_plus_1sigma] = generate_profiles(
        T_samples, Piette
    )
    # ax.plot(T_median, logP, color=color, linewidth=1.5)
    ax.fill_betweenx(
        logP,
        T_minus_1sigma,
        T_plus_1sigma,
        linewidth=0.5,
        color=color,
        facecolor=color,
        alpha=0.5,
        label="95\% confidence interval, {}".format(label),
    )
    logP, MLE_Tprofile = generate_profiles(MLE_Tsample, Piette)
    # ax.plot(T_median, logP, color=color, linewidth=1.5, label="Median profile")
    ax.plot(MLE_Tprofile, logP, color=color, linewidth=4)
    ax.plot(MLE_Tprofile, logP, color=MLE_color, linewidth=2, label="MLE profiles")
    ax.set_ylim([-4, 2.5])
    # reference_TP = true_values[temps]
    # logP, reference_Tprofile = generate_profiles(reference_TP, Piette)
    # ax.plot(reference_Tprofile, logP, color="#444444", linewidth=3)
    # ax.plot(reference_Tprofile, logP, color=reference_color, linewidth=2, label="True profile")
# ax.plot(sonora_T, sonora_P, linewidth=2, linestyle="dashed", color="sandybrown", label=r"SONORA, $\log g$=3.67, $T_\mathrm{eff}$=1584 K", zorder=-8)
# ax.plot(sonora_T, sonora_P, linewidth=4, color="saddlebrown", label="", zorder=-9)
cloud_MLEs = dict(
    list(zip(sample_dicts["2M2236"]["clouds"][1], sample_dicts["2M2236"]["clouds"][3]))
)

fig.gca().invert_yaxis()
ax.set_xlim(ax.get_xlim()[0], 4000)
if cloud_MLEs:
    ax.fill_between(
        np.linspace(ax.get_xlim()[0], 4000),
        cloud_MLEs["Haze_minP"],
        cloud_MLEs["Haze_minP"] + cloud_MLEs["Haze_thick"],
        color="#444444",
        alpha=0.5,
        label="Retrieved cloud layer",
        zorder=-10,
    )

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("log$_{10}$(Pressure/bar)")
handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 0]  # , 4, 5]
# order=[1, 2, 0]
ax.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    fontsize=11,
    facecolor="#444444",
    framealpha=0.25,
    loc="center right",
)

for filetype in filetypes:
    plt.savefig(
        object_label + ".T-P_profiles.{}".format(filetype), bbox_inches="tight", dpi=300
    )

# %%
for group_name in sample_dicts[directory].keys():
    print("Generating corner plots for " + group_name.capitalize())
    for i, (band, sample_dict) in enumerate(sample_dicts.items()):
        if not sample_dict[group_name][0]:
            print("Empty group. Skipping corner plots.")
            continue
        print_names, names, samples, MLE_sample, range, _, lower_error, upper_error = (
            sample_dict[group_name]
        )
        print(samples.shape)
        reference_values = MLE_sample
        # reference_values = reference_dict[group_name][3]
        color = plotting_colors[directories.index(band)]
        plot_generic_legend_first = group_name == "clouds"
        # if i == 0:
        if False:
            base_weights = importance_dicts[band]
            fig, par_titles, title_color = generate_cornerplot(
                samples,
                weights=base_weights,
                group_name=group_name,
                parameter_names=print_names,
                parameter_range=range,
                confidence=0.95,
                color=color,
                MLE_values=MLE_sample,
                MLE_name="Cloudy model",
                MLE_color=color,
                overtext_y=0.8,
                string_formats=group_string_formats[band][group_name],
                # reference_values=reference_values,
                # reference_name=reference_name,
                # reference_markerstyle=reference_markerstyle,
                # reference_color=reference_color,
                reference_values=None,
                reference_name=None,
                reference_markerstyle=None,
                reference_color=None,
                plot_generic_legend_labels=plot_generic_legend_first,
            )
        else:
            print(group_string_formats[band][group_name])
            weights = importance_dicts[band]
            fig, _, _ = generate_cornerplot(
                samples,
                weights=weights,
                group_name=group_name,
                parameter_names=print_names,
                parameter_range=range,
                confidence=0.95,
                # existing_figure=fig,
                # existing_titles=par_titles,
                # existing_title_color=title_color,
                color=color,
                MLE_values=MLE_sample,
                MLE_name="Clear model",
                MLE_color=color,
                string_formats=group_string_formats[band][group_name],
                # reference_values=reference_values,
                # reference_name=reference_name,
                # reference_markerstyle=reference_markerstyle,
                # reference_color=reference_color,
                reference_values=None,
                reference_name=None,
                reference_markerstyle=None,
                reference_color=None,
                plot_generic_legend_labels=True,
            )

    for filetype in filetypes:
        fig.savefig(
            object_label + "." + group_name + ".{}".format(filetype),
            bbox_inches="tight",
            dpi=300,
        )
