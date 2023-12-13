#!/usr/bin/env python3
# Parse output logs of polybench evaluation

import argparse
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import sys
import numpy as np
import scipy.stats as st
from pathlib import Path
from math import sqrt
 
def parse_log_file(log_file):
    found_running = False
    benchmark = ""
    runtime_map = {}

    # parse polybench results
    with open(log_file, 'r') as f:
        for line in f:
            count = 0
            if "Running" in line:
                benchmark = line.split()[1].strip('.exe')
                found_running = True
                if benchmark not in runtime_map:
                    runtime_map[benchmark] = []
                continue

            if found_running:
                runtime = float(line.strip())*1000
                runtime_map[benchmark].append(runtime)
                # add 4 times counter here
                count += 1
                if count == 4:
                    count = 0
                    found_running = False
                continue

    benchmark_mean_time_map = {}
    benchmark_confidence_map = {}
    iterations = -1

    for benchmark,times in sorted(runtime_map.items()):
        # Check number of iterations of each benchmark
        if iterations == -1:
            iterations = len(times)
        else:
            # modify times 4 here
            assert(iterations == len(times))
        mean_time = np.mean(times)
        conf_low, conf_high = st.norm.interval(alpha=0.95, loc=mean_time, scale=st.sem(times))
        conf = (conf_high - conf_low) / 2
        benchmark_mean_time_map[benchmark] = mean_time
        benchmark_confidence_map[benchmark] = conf

    return benchmark_mean_time_map, benchmark_confidence_map, iterations

def parse_log_baseline_file(log_file):
    found_running = False
    benchmark = ""
    runtime_map = {}

    # parse polybench results
    with open(log_file, 'r') as f:
        for line in f:
            count = 0
            if "Running" in line:
                benchmark = line.split()[1].strip('.exe')
                found_running = True
                if benchmark not in runtime_map:
                    runtime_map[benchmark] = []
                    runtime_map[benchmark].append(float(0.0))
                else:
                    runtime_map[benchmark].append(float(0.0))
                continue
            if found_running:
                runtime = float(line.strip())*1000
                runtime_map[benchmark][-1] += runtime
                # add 4 times counter here
                count += 1
                if count == 4:
                    count = 0
                    found_running = False
                continue

    benchmark_mean_time_map = {}
    benchmark_confidence_map = {}
    iterations = -1

    for benchmark,times in sorted(runtime_map.items()):
        # Check number of iterations of each benchmark
        if iterations == -1:
            iterations = len(times)
        else:
            # modify times 4 here
            assert(iterations == len(times))
        mean_time = np.mean(times)
        conf_low, conf_high = st.norm.interval(alpha=0.95, loc=mean_time, scale=st.sem(times))
        conf = (conf_high - conf_low) / 2
        benchmark_mean_time_map[benchmark] = mean_time
        benchmark_confidence_map[benchmark] = conf

    return benchmark_mean_time_map, benchmark_confidence_map, iterations

# parse perf results
def parse_perf_log_file(log_file):
    # map from benchmark to perf counter to a list of counter values
    perf_counter_map = {}

    with open(log_file, 'r') as f:
        collectPerf = False
        for line in f:
            # skip empty lines
            if line.strip() == '':
                continue

            if "Running" in line:
                benchmark = line.split()[1].strip('.exe')
                if benchmark not in perf_counter_map:
                    perf_counter_map[benchmark] = {}
            # start of perf information
            elif "Performance counter stats" in line:
                collectPerf = True
            # end of perf information
            elif "seconds time elapsed" in line:
                collectPerf = False
            elif collectPerf:
                value = int(line.split()[0].replace(',',''))
                counter = line.split()[1]
                if counter not in perf_counter_map[benchmark]:
                    perf_counter_map[benchmark][counter] = []
                perf_counter_map[benchmark][counter].append(value)

    perf_mean_map = {}
    perf_confidence_map = {}

    for benchmark in sorted(perf_counter_map.keys()):
        for counter in sorted(perf_counter_map[benchmark].keys()):
            values = perf_counter_map[benchmark][counter]
            mean_time = np.mean(values)
            conf_low, conf_high = st.norm.interval(alpha=0.95, loc=mean_time, scale=st.sem(values))
            conf = (conf_high - conf_low) / 2
            if benchmark not in perf_mean_map:
                perf_mean_map[benchmark] = {}
                perf_confidence_map[benchmark] = {}
            perf_mean_map[benchmark][counter] = mean_time
            perf_confidence_map[benchmark][counter] = conf

    return perf_mean_map, perf_confidence_map

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse output logs of polybench evaluation.")

    parser.add_argument("input_dir", help="Log dir obtained with run.sh")
    parser.add_argument("output_dir", help="Output dir")
    parser.add_argument("tiling_method", help="Tiling method used in generate-files.sh.", type=str, choices=['AffineTiling', 'Polymer'])
    parser.add_argument("--skip-perf-graphs", help="Do not generate perf graphs.", action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tiling_method = args.tiling_method
    skip_perf_graphs = args.skip_perf_graphs

    if not input_dir.exists() or not input_dir.is_dir():
        print("Input file is not a file of does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not output_dir.exists() or not output_dir.is_dir():
        print("Output dir does not exist", file=sys.stderr)
        sys.exit(1)

    input_dir = input_dir.absolute()
    output_dir = output_dir.absolute()

    # from matplotlib import rc
    # rc('font',**{'family':'serif','serif':['Libertine']})
    # rc('text', usetex=True)
    # rc('text.latex', 
    #    preamble="\n".join([ # plots will use this preamble
    #     r"\usepackage[utf8]{inputenc}",
    #     r"\usepackage[T1]{fontenc}",
    #     r"\usepackage{libertine}",
    #     ])
    #   )
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        "font.size": 16,
        "font.family": "Dejavu Serif",
        "legend.fontsize": 14,
    })
    marker=['o', 'v', '^', '<', '>', 's', 'p', '*', 'X']

    if tiling_method == "AffineTiling":
    
        polly_log = input_dir / "polly.log"
        polygeist_log = input_dir / "polygeist.log"

        polly_mean, polly_conf, _ = parse_log_file(polly_log)
        polygeist_mean, polygeist_conf, _ = parse_log_file(polygeist_log)

        affine_tiling_mean_list = {}
        affine_tiling_conf_list = {}
        affine_tiling_packing_mean_list = {}
        affine_tiling_packing_conf_list = {}
        for level in ('l1', 'l2', 'l3'):
            affine_tiling_mean, affine_tiling_conf, _ = parse_log_file(input_dir / "affine-tiling-{}.log".format(level))
            affine_tiling_mean_list[level] = affine_tiling_mean
            affine_tiling_conf_list[level] = affine_tiling_conf

            affine_tiling_packing_mean, affine_tiling_packing_conf, _ = parse_log_file(input_dir / "affine-tiling-{}-packing.log".format(level))
            affine_tiling_packing_mean_list[level] = affine_tiling_packing_mean
            affine_tiling_packing_conf_list[level] = affine_tiling_packing_conf

        # Build bar graph for execution time ------------------------------------------------------
        for benchmark in polly_mean.keys():
            y_time_affine_tiling = []
            y_conf_affine_tiling = []
            x_cache_level = []
            y_time_affine_tiling_packing = []
            y_conf_affine_tiling_packing = []
            x_cache_level_packings = []
            for level in sorted(affine_tiling_packing_mean_list.keys()):
                if benchmark in affine_tiling_packing_mean_list[level]:
                    y_time_affine_tiling_packing.append(affine_tiling_packing_mean_list[level][benchmark])
                    y_conf_affine_tiling_packing.append(affine_tiling_packing_conf_list[level][benchmark])
                    x_cache_level_packings.append(level)
                    y_time_affine_tiling.append(affine_tiling_mean_list[level][benchmark])
                    y_conf_affine_tiling.append(affine_tiling_conf_list[level][benchmark])
                    x_cache_level.append(level)

            x = [int(l[-1])-1 for l in x_cache_level]
            x = np.array(x)  # the label locations
            width = 0.2  # the width of the bars

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width/2, y_time_affine_tiling, width, yerr=y_conf_affine_tiling, color='#b2abd2', edgecolor='black', linewidth=0.5, alpha=0.8, label='Affine')
            rects2 = ax.bar(x + width/2, y_time_affine_tiling_packing, width, yerr=y_conf_affine_tiling_packing, color='#e66101', edgecolor='black', linewidth=0.25, alpha=0.8, label='Affine + GPAT')

            plt.axhline(y=polly_mean[benchmark], color='black', linestyle='-', alpha=0.8, linewidth=1, label='Polly')
            ax.fill_between([-0.5, 2.5], polly_mean[benchmark]+polly_conf[benchmark], polly_mean[benchmark]-polly_conf[benchmark], alpha=0.3, color='#000000')
            plt.axhline(y=polygeist_mean[benchmark], color='black', linestyle='--', alpha=0.8, linewidth=1, label='Clang-O3')
            ax.fill_between([-0.5, 2.5], polygeist_mean[benchmark]+polygeist_conf[benchmark], polygeist_mean[benchmark]-polygeist_conf[benchmark], alpha=0.3, color='#000000')

            # Set axes labels and limits
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=-0.5, right=2.5)
            ax.set_xticks(x, [l.upper() for l in x_cache_level])
            ax.xaxis.grid(True)
            ax.grid(which='both', alpha=0.3)
            ax.set_ylabel('CPU Time (ms)')
            ax.set_xlabel('Tiling target cache level')
            # Save the figure and show
            graph_path = output_dir / (benchmark + '-time.png')
            legend = plt.legend(ncol=2, loc='lower center', frameon=True, framealpha=1, bbox_to_anchor=(0.5, 1), columnspacing=0.8, handletextpad=0.3, handlelength=1.0)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('black')
            plt.savefig(graph_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        # -----------------------------------------------------------------------------------------

        # Build bar graph for speedup -------------------------------------------------------------
        for benchmark in polly_mean.keys():
            y_time_affine_tiling = []
            y_conf_affine_tiling = []
            x_cache_level = []
            y_time_affine_tiling_packing = []
            y_conf_affine_tiling_packing = []
            x_cache_level_packings = []
            for level in sorted(affine_tiling_packing_mean_list.keys()):
                if benchmark in affine_tiling_packing_mean_list[level]:
                    y_time_affine_tiling_packing.append(polygeist_mean[benchmark]/affine_tiling_packing_mean_list[level][benchmark])
                    # y_conf_affine_tiling_packing.append(affine_tiling_packing_conf_list[level][benchmark])
                    x_cache_level_packings.append(level)
                    y_time_affine_tiling.append(polygeist_mean[benchmark]/affine_tiling_mean_list[level][benchmark])
                    # y_conf_affine_tiling.append(affine_tiling_conf_list[level][benchmark])
                    x_cache_level.append(level)

            x = [int(l[-1])-1 for l in x_cache_level]
            x = np.array(x)  # the label locations
            width = 0.2  # the width of the bars

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width/2, y_time_affine_tiling, width, color='#b2abd2', edgecolor='black', linewidth=0.5, alpha=0.8, label='Affine')
            rects2 = ax.bar(x + width/2, y_time_affine_tiling_packing, width, color='#e66101', edgecolor='black', linewidth=0.25, alpha=0.8, label='Affine + GPAT')

            plt.axhline(y=polygeist_mean[benchmark]/polly_mean[benchmark], color='black', linestyle='-', alpha=0.8, linewidth=1, label='Polly')
            ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)

            # Set axes labels and limits
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=-0.5, right=2.5)
            ax.set_xticks(x, [l.upper() for l in x_cache_level])
            ax.xaxis.grid(True)
            ax.grid(which='both', alpha=0.3)
            ax.set_ylabel('Speedup over Clang-O3')
            ax.set_xlabel('Tiling target cache level')
            # Save the figure and show
            graph_path = output_dir / (benchmark + '-speedup.png')
            legend = plt.legend(ncol=3, loc='lower center', frameon=True, framealpha=1, bbox_to_anchor=(0.5, 1), columnspacing=0.8, handletextpad=0.3, handlelength=1.0)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('black')
            plt.savefig(graph_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        # -----------------------------------------------------------------------------------------

        # Build combined bar graph for speedup for paper ------------
        fig, ax = plt.subplots()
        base_x = 1
        x_positions = []
        x_ticks = []
        x_ticks_positions = []
        y_colors = []
        y_pattern = []
        y_times = []
        # y_confs = []

        for benchmark in ["2mm", "3mm", "doitgen", "contraction-3d", "gramschmidt", "trmm"]:
            for level in sorted(affine_tiling_packing_mean_list.keys()):
                if benchmark in affine_tiling_packing_mean_list[level]:
                    y_times.append(polygeist_mean[benchmark]/affine_tiling_mean_list[level][benchmark])
                    y_colors.append('#b2abd2')
                    y_pattern.append("")
                    # y_conf_affine_tiling.append(affine_tiling_conf_list[level][benchmark])
                    y_times.append(polygeist_mean[benchmark]/affine_tiling_packing_mean_list[level][benchmark])
                    y_colors.append('#e66101')
                    y_pattern.append("")
                    # y_conf_affine_tiling_packing.append(affine_tiling_packing_conf_list[level][benchmark])
                    # there is no tiling in gramschmidt
                    if benchmark == "gramschmidt":
                        break
            y_times.append(polygeist_mean[benchmark]/polly_mean[benchmark])
            y_colors.append('#000000')
            y_pattern.append("")

            # the width of the bars
            width = 2

            # set positions for all bars
            if benchmark in ["2mm", "3mm"]:
                x_positions += [base_x, base_x+width, base_x+1+2*width, base_x+1+3*width, base_x+2+4*width]
            elif benchmark == "doitgen":
                x_positions += [base_x, base_x+width, base_x+1+2*width, base_x+1+3*width, base_x+2+4*width, base_x+2+5*width, base_x+3+6*width]
            elif benchmark in ["gramschmidt", "trmm", "contraction-3d"]:
                x_positions += [base_x, base_x+width, base_x+1+2*width] 

            # set positions for all bars
            if benchmark in ["2mm", "3mm"]:
                x_ticks += ["L1", "L2"]
                x_ticks_positions += [(base_x+base_x+width)/2, (base_x+1+2*width+base_x+1+3*width)/2]
            elif benchmark == "doitgen":
                x_ticks += ["L1", "L2", "L3"]
                x_ticks_positions += [(base_x+base_x+width)/2, (base_x+1+2*width+base_x+1+3*width)/2, (base_x+2+4*width+base_x+2+5*width)/2]
            elif benchmark in ["gramschmidt", "contraction-3d"]:
                x_ticks += ["X"]
                x_ticks_positions += [(base_x+base_x+width)/2]
            elif benchmark == "trmm":
                x_ticks += ["L3"]
                x_ticks_positions += [(base_x+base_x+width)/2]

            if benchmark in ["2mm", "3mm"]:
                base_x += 2+5*width + 2*width
            elif benchmark == "doitgen":
                base_x += 3+7*width + 2*width
            elif benchmark in ["gramschmidt", "trmm", "contraction-3d"]:
                base_x += 12

        for position,time,color,pattern in zip(x_positions,y_times,y_colors,y_pattern):
            ax.bar(position, time, width, color=color, edgecolor='black', linewidth=0.5, alpha=1)

        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)

        # Set axes labels and limits
        ax.set_ylim(bottom=0)
        ax.set_xticks(x_ticks_positions, x_ticks)
        ax.tick_params(axis='x', which='both', labelsize=14)
        ax.xaxis.grid(True)
        ax.grid(which='both', alpha=0.3)
        ax.set_ylabel('Speedup over Clang-O3')
        # ax.set_xlabel('Affine target cache level')
        # ax.xaxis.set_label_coords(0.5, -0.25)
        # Save the figure and show
        graph_path = output_dir / 'all-bars-speedup.png'

        affine_patch = mpatches.Patch(facecolor='#b2abd2', label='Affine', alpha=1, edgecolor="black", linewidth=0.5)
        affine_packing_patch = mpatches.Patch(facecolor='#e66101', label='Affine + GPAT', alpha=1, edgecolor="black", linewidth=0.5)
        polly_patch = mpatches.Patch(facecolor='#000000', label='Polly', alpha=1, edgecolor="black", linewidth=0.5)

        legend = plt.legend(ncol=2, loc='lower center', frameon=True, framealpha=1, bbox_to_anchor=(0.5, 1), handles=[affine_patch, affine_packing_patch, polly_patch], handletextpad=0.3, handlelength=1.0)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')

        # ax.text((x_positions[0]+x_positions[4]+3*width)/2, -0.5, "{\\em \\rmfamily 2mm}", rotation=45, va="top", ha="right")
        # ax.text((x_positions[5]+x_positions[9]+3*width)/2, -0.5, "{\\em \\rmfamily 3mm}", rotation=45, va="top", ha="right")
        # ax.text((x_positions[10]+x_positions[16]+3*width)/2, -0.5, "{\\em \\rmfamily doitgen}", rotation=45, va="top", ha="right")
        # ax.text((x_positions[17]+x_positions[19]+3*width)/2, -0.5, "{\\em \\rmfamily contract3D}", rotation=45, va="top", ha="right")
        # ax.text((x_positions[20]+x_positions[22]+3*width)/2, -0.5, "{\\em \\rmfamily gramschmidt}", rotation=45, va="top", ha="right")
        # ax.text((x_positions[23]+x_positions[25]+3*width)/2, -0.5, "{\\em \\rmfamily trmm}", rotation=45, va="top", ha="right")

        ax.text((x_positions[0]+x_positions[4]+3*width)/2, -0.5, "2mm", rotation=45, va="top", ha="right")
        ax.text((x_positions[5]+x_positions[9]+3*width)/2, -0.5, "3mm", rotation=45, va="top", ha="right")
        ax.text((x_positions[10]+x_positions[16]+3*width)/2, -0.5, "doitgen", rotation=45, va="top", ha="right")
        ax.text((x_positions[17]+x_positions[19]+3*width)/2, -0.5, "contract3D", rotation=45, va="top", ha="right")
        ax.text((x_positions[20]+x_positions[22]+3*width)/2, -0.5, "gramschmidt", rotation=45, va="top", ha="right")
        ax.text((x_positions[23]+x_positions[25]+3*width)/2, -0.5, "trmm", rotation=45, va="top", ha="right")
        
        plt.savefig(graph_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        # -----------------------------------------------------------------------------------------
    
    elif tiling_method == "Polymer":

        plt.rcParams.update({
            "font.size": 18,
            "legend.fontsize": 16,
        })

        polly_log = input_dir / "polly.log"
        polygeist_log = input_dir / "polygeist.log"
        polymer_logs = input_dir.glob("polymer-[0-9]*.log")
        polymer_packing_logs = input_dir.glob("polymer-packing-[0-9]*.log")

        polly_mean, polly_conf, _ = parse_log_file(polly_log)
        polygeist_mean, polygeist_conf, _ = parse_log_file(polygeist_log)

        tiling_polymer_mean_map = {}
        tiling_polymer_conf_map = {}
        for log_file in polymer_logs:
            tiling_size = int(log_file.name.strip('.log').split('-')[1])
            polymer_mean, polymer_conf, _ = parse_log_file(log_file)
            tiling_polymer_mean_map[tiling_size] = polymer_mean
            tiling_polymer_conf_map[tiling_size] = polymer_conf

        tiling_polymer_packing_mean_map = {}
        tiling_polymer_packing_conf_map = {}
        for log_file in polymer_packing_logs:
            tiling_size = int(log_file.name.strip('.log').split('-')[2])
            polymer_packing_mean, polymer_packing_conf, _ = parse_log_file(log_file)
            tiling_polymer_packing_mean_map[tiling_size] = polymer_packing_mean
            tiling_polymer_packing_conf_map[tiling_size] = polymer_packing_conf

        x_tilings = []
        for tiling in sorted(tiling_polymer_mean_map.keys()):
            x_tilings.append(tiling)

        # Build time graphs ---------------------------------------------------------------------------
        for benchmark in polly_mean.keys():
            y_time_polymer = []
            y_conf_polymer = []
            for tiling in sorted(tiling_polymer_mean_map.keys()):
                y_time_polymer.append(tiling_polymer_mean_map[tiling][benchmark])
                y_conf_polymer.append(tiling_polymer_conf_map[tiling][benchmark])

            y_time_polymer_packing = []
            y_conf_polymer_packing = []
            x_tilings_polymer_packing = []
            for tiling in sorted(tiling_polymer_packing_mean_map.keys()):
                if benchmark in tiling_polymer_packing_mean_map[tiling]:
                    y_time_polymer_packing.append(tiling_polymer_packing_mean_map[tiling][benchmark])
                    y_conf_polymer_packing.append(tiling_polymer_packing_conf_map[tiling][benchmark])
                    x_tilings_polymer_packing.append(tiling)

            fig, ax = plt.subplots()
            plt.axhline(y=polly_mean[benchmark], color='black', linestyle='-', alpha=0.8, linewidth=1, label='Polly')
            ax.fill_between(x_tilings, polly_mean[benchmark]+polly_conf[benchmark], polly_mean[benchmark]-polly_conf[benchmark], alpha=0.3, color='#000000')
            plt.axhline(y=polygeist_mean[benchmark], color='black', linestyle='--', alpha=0.8, linewidth=1, label='Clang-O3')
            ax.fill_between(x_tilings, polygeist_mean[benchmark]+polygeist_conf[benchmark], polygeist_mean[benchmark]-polygeist_conf[benchmark], alpha=0.3, color='#000000')
            ax.errorbar(x_tilings, y_time_polymer, yerr=y_conf_polymer, label="Polymer", markersize=sqrt(18), markerfacecolor='#5e3c99', markeredgecolor='black', markeredgewidth=0.5, ecolor='black', elinewidth=0.5, fmt='o', alpha=0.8)
            ax.errorbar(x_tilings_polymer_packing, y_time_polymer_packing, yerr=y_conf_polymer_packing, label="Polymer + GPAT", markersize=sqrt(18), markerfacecolor='#e66101', markeredgecolor='black', markeredgewidth=0.5, ecolor='black', elinewidth=0.5, fmt='s', alpha=0.8)
            # Set axes labels and limits
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=min(x_tilings)-2, right=max(x_tilings)+2)
            ax.yaxis.grid(True)
            ax.xaxis.grid(True)
            ax.grid(which='both', alpha=0.3)
            ax.set_ylabel('CPU Time (ms)')
            ax.set_xlabel('Tiling size (all dimensions)')
            # Save the figure and show
            graph_path = output_dir / (benchmark + '-time.png')
            legend = plt.legend(ncol=3, loc='lower center', frameon=True, framealpha=1, bbox_to_anchor=(0.5, 1), columnspacing=0.8, handletextpad=0.3, handlelength=1.0)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('black')
            plt.savefig(graph_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        # ---------------------------------------------------------------------------------------------

        # Build speedup graphs ------------------------------------------------------------------------
        for benchmark in polly_mean.keys():
            y_time_polymer = []
            y_conf_polymer = []
            for tiling in sorted(tiling_polymer_mean_map.keys()):
                y_time_polymer.append(polygeist_mean[benchmark]/tiling_polymer_mean_map[tiling][benchmark])

            y_time_polymer_packing = []
            y_conf_polymer_packing = []
            x_tilings_polymer_packing = []
            for tiling in sorted(tiling_polymer_packing_mean_map.keys()):
                if benchmark in tiling_polymer_packing_mean_map[tiling]:
                    y_time_polymer_packing.append(polygeist_mean[benchmark]/tiling_polymer_packing_mean_map[tiling][benchmark])
                    x_tilings_polymer_packing.append(tiling)

            fig, ax = plt.subplots()
            plt.axhline(y=polygeist_mean[benchmark]/polly_mean[benchmark], color='black', linestyle='-', alpha=0.8, linewidth=1, label='Polly')
            plt.axhline(y=1, color='black', linestyle='--', alpha=1, linewidth=1)
            ax.errorbar(x_tilings, y_time_polymer, label="Polymer", markersize=sqrt(18), markerfacecolor='#5e3c99', markeredgecolor='black', markeredgewidth=0.5, ecolor='black', elinewidth=0.5, fmt='o', alpha=1)
            ax.errorbar(x_tilings_polymer_packing, y_time_polymer_packing, label="Polymer + GPAT", markersize=sqrt(18), markerfacecolor='#e66101', markeredgecolor='black', markeredgewidth=0.5, ecolor='black', elinewidth=0.5, fmt='s', alpha=0.8)
            # Set axes labels and limits
            ax.set_xlim(left=min(x_tilings)-2, right=max(x_tilings)+2)
            ax.yaxis.grid(True)
            ax.xaxis.grid(True)
            ax.grid(which='both', alpha=0.3)
            ax.set_ylabel('Speedup over Clang-O3')
            ax.set_xlabel('Tiling size (all dimensions)')
            # Save the figure and show
            graph_path = output_dir / (benchmark + '-speedup.png')
            legend = plt.legend(ncol=3, loc='lower center', frameon=True, framealpha=1, bbox_to_anchor=(0.5, 1), columnspacing=0.8, handletextpad=0.3, handlelength=1.0)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('black')
            plt.savefig(graph_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        # ---------------------------------------------------------------------------------------------

    # Get perf counters that were measured
    perf_counters = set()
    # get perf counters that were measured
    with open(input_dir / "polly.log", 'r') as f:
        collectPerf = False
        for line in f:
            # skip emtpy lines
            if line.strip() == '':
                continue

            # start of perf information
            if "Performance counter stats" in line:
                collectPerf = True
            # end of perf information
            elif "seconds time elapsed" in line:
                break
            elif collectPerf:
                perf_counters.add(line.split()[1])

    perf_found = True
    if skip_perf_graphs:
        perf_found = False
    # Check if perf was measured
    elif len(perf_counters) == 0:
        perf_found = False
        print("Warning: No perf data was found")

    if perf_found:
        # Output paths
        perf_outputs_dir = output_dir / "perf"
        perf_relative_outputs_dir = output_dir / "perf-relative"

        perf_outputs_dir.mkdir(exist_ok=True)
        perf_relative_outputs_dir.mkdir(exist_ok=True)

        if tiling_method == "AffineTiling":
            polly_log = input_dir / "polly.log"
            polygeist_log = input_dir / "polygeist.log"

            polly_perf_mean, polly_perf_conf = parse_perf_log_file(polly_log)
            polygeist_perf_mean, polygeist_perf_conf = parse_perf_log_file(polygeist_log)

            affine_tiling_perf_mean_list = {}
            affine_tiling_perf_conf_list = {}
            affine_tiling_packing_perf_mean_list = {}
            affine_tiling_packing_perf_conf_list = {}
            for level in ('l1', 'l2', 'l3'):
                affine_tiling_mean, affine_tiling_conf = parse_perf_log_file(input_dir / "affine-tiling-{}.log".format(level))
                affine_tiling_perf_mean_list[level] = affine_tiling_mean
                affine_tiling_perf_conf_list[level] = affine_tiling_conf

                affine_tiling_packing_mean, affine_tiling_packing_conf = parse_perf_log_file(input_dir / "affine-tiling-{}-packing.log".format(level))
                affine_tiling_packing_perf_mean_list[level] = affine_tiling_packing_mean
                affine_tiling_packing_perf_conf_list[level] = affine_tiling_packing_conf

            # Build bar graph for execution time ------------------------------------------------------
            for benchmark in polly_perf_mean.keys():
                for counter in perf_counters:
                    y_time_affine_tiling = []
                    y_conf_affine_tiling = []
                    x_cache_level = []
                    y_time_affine_tiling_packing = []
                    y_conf_affine_tiling_packing = []
                    x_cache_level_packings = []
                    for level in sorted(affine_tiling_packing_perf_mean_list.keys()):
                        if benchmark in affine_tiling_packing_perf_mean_list[level]:
                            y_time_affine_tiling_packing.append(affine_tiling_packing_perf_mean_list[level][benchmark][counter])
                            y_conf_affine_tiling_packing.append(affine_tiling_packing_perf_conf_list[level][benchmark][counter])
                            x_cache_level_packings.append(level)
                            y_time_affine_tiling.append(affine_tiling_perf_mean_list[level][benchmark][counter])
                            y_conf_affine_tiling.append(affine_tiling_perf_conf_list[level][benchmark][counter])
                            x_cache_level.append(level)

                    x = [int(l[-1])-1 for l in x_cache_level]
                    x = np.array(x)  # the label locations
                    width = 0.2  # the width of the bars

                    fig, ax = plt.subplots()
                    rects1 = ax.bar(x - width/2, y_time_affine_tiling, width, yerr=y_conf_affine_tiling, color='#b2abd2', edgecolor='black', linewidth=0.5, alpha=0.8, label='Affine')
                    rects2 = ax.bar(x + width/2, y_time_affine_tiling_packing, width, yerr=y_conf_affine_tiling_packing, color='#e66101', edgecolor='black', linewidth=0.25, alpha=0.8, label='Affine + GPAT')

                    plt.axhline(y=polly_perf_mean[benchmark][counter], color='black', linestyle='-', alpha=0.8, linewidth=1, label='Polly')
                    ax.fill_between([-0.5, 2.5], polly_perf_mean[benchmark][counter]+polly_perf_conf[benchmark][counter], polly_perf_mean[benchmark][counter]-polly_perf_conf[benchmark][counter], alpha=0.3, color='#000000')
                    plt.axhline(y=polygeist_perf_mean[benchmark][counter], color='black', linestyle='--', alpha=0.8, linewidth=1, label='Clang-O3')
                    ax.fill_between([-0.5, 2.5], polygeist_perf_mean[benchmark][counter]+polygeist_perf_conf[benchmark][counter], polygeist_perf_mean[benchmark][counter]-polygeist_perf_conf[benchmark][counter], alpha=0.3, color='#000000')

                    # Set axes labels and limits
                    ax.set_ylim(bottom=0)
                    ax.set_xlim(left=-0.5, right=2.5)
                    ax.set_xticks(x, [l.upper() for l in x_cache_level])
                    ax.xaxis.grid(True)
                    ax.grid(which='both', alpha=0.3)
                    ax.set_ylabel(counter)
                    ax.set_xlabel('Tiling target cache level')
                    # Save the figure and show
                    graph_path = perf_outputs_dir / (benchmark + '-' + counter + '.png')
                    legend = plt.legend(ncol=2, loc='lower center', frameon=True, framealpha=1, bbox_to_anchor=(0.5, 1), columnspacing=0.8, handletextpad=0.3, handlelength=1.0)
                    frame = legend.get_frame()
                    frame.set_facecolor('white')
                    frame.set_edgecolor('black')
                    plt.savefig(graph_path, bbox_inches='tight', dpi=300)
                    plt.close(fig)
            # -----------------------------------------------------------------------------------------

            # Build bar graph for speedup -------------------------------------------------------------
            for benchmark in polly_perf_mean.keys():
                for counter in perf_counters:
                    y_time_affine_tiling = []
                    y_conf_affine_tiling = []
                    x_cache_level = []
                    y_time_affine_tiling_packing = []
                    y_conf_affine_tiling_packing = []
                    x_cache_level_packings = []
                    for level in sorted(affine_tiling_packing_perf_mean_list.keys()):
                        if benchmark in affine_tiling_packing_perf_mean_list[level]:
                            y_time_affine_tiling_packing.append(polygeist_perf_mean[benchmark][counter]/affine_tiling_packing_perf_mean_list[level][benchmark][counter])
                            x_cache_level_packings.append(level)
                            y_time_affine_tiling.append(polygeist_perf_mean[benchmark][counter]/affine_tiling_perf_mean_list[level][benchmark][counter])
                            x_cache_level.append(level)

                    x = [int(l[-1])-1 for l in x_cache_level]
                    x = np.array(x)  # the label locations
                    width = 0.2  # the width of the bars

                    fig, ax = plt.subplots()
                    rects1 = ax.bar(x - width/2, y_time_affine_tiling, width, color='#b2abd2', edgecolor='black', linewidth=0.5, alpha=0.8, label='Affine')
                    rects2 = ax.bar(x + width/2, y_time_affine_tiling_packing, width, color='#e66101', edgecolor='black', linewidth=0.25, alpha=0.8, label='Affine + GPAT')

                    plt.axhline(y=polygeist_perf_mean[benchmark][counter]/polly_perf_mean[benchmark][counter], color='black', linestyle='-', alpha=0.8, linewidth=1, label='Polly')
                    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)

                    # Set axes labels and limits
                    ax.set_ylim(bottom=0)
                    ax.set_xlim(left=-0.5, right=2.5)
                    ax.set_xticks(x, [l.upper() for l in x_cache_level])
                    ax.xaxis.grid(True)
                    ax.grid(which='both', alpha=0.3)
                    ax.set_ylabel(counter + '\nreduction over Clang-O3')
                    ax.set_xlabel('Tiling target cache level')
                    # Save the figure and show
                    graph_path = perf_relative_outputs_dir / (benchmark + '-' + counter + '.png')
                    legend = plt.legend(ncol=3, loc='lower center', frameon=True, framealpha=1, bbox_to_anchor=(0.5, 1), columnspacing=0.8, handletextpad=0.3, handlelength=1.0)
                    frame = legend.get_frame()
                    frame.set_facecolor('white')
                    frame.set_edgecolor('black')
                    plt.savefig(graph_path, bbox_inches='tight', dpi=300)
                    plt.close(fig)
            # -----------------------------------------------------------------------------------------

        elif tiling_method == "Polymer":
            plt.rcParams.update({
                "font.size": 18,
                "legend.fontsize": 16,
            })

            polly_log = input_dir / "polly.log"
            polygeist_log = input_dir / "polygeist.log"
            polymer_logs = input_dir.glob("polymer-[0-9]*.log")
            polymer_packing_logs = input_dir.glob("polymer-packing-[0-9]*.log")

            polly_perf_mean, polly_perf_conf = parse_perf_log_file(polly_log)
            polygeist_perf_mean, polygeist_perf_conf = parse_perf_log_file(polygeist_log)

            tiling_polymer_perf_mean_map = {}
            tiling_polymer_perf_conf_map = {}
            for log_file in polymer_logs:
                tiling_size = int(log_file.name.strip('.log').split('-')[1])
                polymer_perf_mean, polymer_perf_conf = parse_perf_log_file(log_file)
                tiling_polymer_perf_mean_map[tiling_size] = polymer_perf_mean
                tiling_polymer_perf_conf_map[tiling_size] = polymer_perf_conf

            tiling_polymer_packing_perf_mean_map = {}
            tiling_polymer_packing_perf_conf_map = {}
            for log_file in polymer_packing_logs:
                tiling_size = int(log_file.name.strip('.log').split('-')[2])
                polymer_packing_perf_mean, polymer_packing_perf_conf = parse_perf_log_file(log_file)
                tiling_polymer_packing_perf_mean_map[tiling_size] = polymer_packing_perf_mean
                tiling_polymer_packing_perf_conf_map[tiling_size] = polymer_packing_perf_conf

            x_tilings = []
            for tiling in sorted(tiling_polymer_perf_mean_map.keys()):
                x_tilings.append(tiling)

            # Build perf graphs ---------------------------------------------------------------------------
            for benchmark in polly_mean.keys():
                for counter in perf_counters:
                    y_value_polymer = []
                    y_conf_polymer = []
                    for tiling in sorted(tiling_polymer_perf_mean_map.keys()):
                        y_value_polymer.append(tiling_polymer_perf_mean_map[tiling][benchmark][counter])
                        y_conf_polymer.append(tiling_polymer_perf_conf_map[tiling][benchmark][counter])

                    y_value_polymer_packing = []
                    y_conf_polymer_packing = []
                    x_tilings_polymer_packing = []
                    for tiling in sorted(tiling_polymer_packing_perf_mean_map.keys()):
                        if benchmark in tiling_polymer_packing_perf_mean_map[tiling]:
                            y_value_polymer_packing.append(tiling_polymer_packing_perf_mean_map[tiling][benchmark][counter])
                            y_conf_polymer_packing.append(tiling_polymer_packing_perf_conf_map[tiling][benchmark][counter])
                            x_tilings_polymer_packing.append(tiling)

                    fig, ax = plt.subplots()
                    plt.axhline(y=polly_perf_mean[benchmark][counter], color='black', linestyle='-', alpha=0.8, linewidth=1, label='Polly')
                    ax.fill_between(x_tilings, polly_perf_mean[benchmark][counter]+polly_perf_conf[benchmark][counter], polly_perf_mean[benchmark][counter]-polly_perf_conf[benchmark][counter], alpha=0.3, color='#000000')
                    plt.axhline(y=polygeist_perf_mean[benchmark][counter], color='black', linestyle='--', alpha=0.8, linewidth=1, label='Clang-O3')
                    ax.fill_between(x_tilings, polygeist_perf_mean[benchmark][counter]+polygeist_perf_conf[benchmark][counter], polygeist_perf_mean[benchmark][counter]-polygeist_perf_conf[benchmark][counter], alpha=0.3, color='#000000')
                    ax.errorbar(x_tilings, y_value_polymer, yerr=y_conf_polymer, label="Polymer", markersize=sqrt(18), markerfacecolor='#5e3c99', markeredgecolor='black', markeredgewidth=0.5, ecolor='black', elinewidth=0.5, fmt='o', alpha=0.8)
                    ax.errorbar(x_tilings_polymer_packing, y_value_polymer_packing, yerr=y_conf_polymer_packing, label="Polymer + GPAT", markersize=sqrt(18), markerfacecolor='#e66101', markeredgecolor='black', markeredgewidth=0.5, ecolor='black', elinewidth=0.5, fmt='s', alpha=0.8)
                    # Set axes labels and limits
                    ax.set_ylim(bottom=0)
                    ax.set_xlim(left=min(x_tilings)-2, right=max(x_tilings)+2)
                    ax.yaxis.grid(True)
                    ax.xaxis.grid(True)
                    ax.grid(which='both', alpha=0.3)
                    ax.set_ylabel(counter)
                    ax.set_xlabel('Tiling size (all dimensions)')
                    # Save the figure and show
                    graph_path = perf_outputs_dir / (benchmark + '-' + counter + '.png')
                    legend = plt.legend(ncol=2, loc='lower center', frameon=True, framealpha=1, bbox_to_anchor=(0.5, 1), columnspacing=0.8, handletextpad=0.3, handlelength=1.0)
                    frame = legend.get_frame()
                    frame.set_facecolor('white')
                    frame.set_edgecolor('black')
                    plt.savefig(graph_path, bbox_inches='tight', dpi=300)
                    plt.close(fig)
                # ---------------------------------------------------------------------------------------------

            # Build perf relative graphs ---------------------------------------------------------------------------
            for benchmark in polly_mean.keys():
                for counter in perf_counters:
                    y_value_polymer = []
                    for tiling in sorted(tiling_polymer_perf_mean_map.keys()):
                        y_value_polymer.append(polygeist_perf_mean[benchmark][counter]/tiling_polymer_perf_mean_map[tiling][benchmark][counter])

                    y_value_polymer_packing = []
                    x_tilings_polymer_packing = []
                    for tiling in sorted(tiling_polymer_packing_perf_mean_map.keys()):
                        if benchmark in tiling_polymer_packing_perf_mean_map[tiling]:
                            y_value_polymer_packing.append(polygeist_perf_mean[benchmark][counter]/tiling_polymer_packing_perf_mean_map[tiling][benchmark][counter])
                            x_tilings_polymer_packing.append(tiling)

                    fig, ax = plt.subplots()
                    plt.axhline(y=polygeist_perf_mean[benchmark][counter]/polly_perf_mean[benchmark][counter], color='black', linestyle='-', alpha=0.8, linewidth=1, label='Polly')
                    plt.axhline(y=1, color='black', linestyle='--', alpha=0.8, linewidth=1)
                    ax.errorbar(x_tilings, y_value_polymer, label="Polymer", markersize=sqrt(18), markerfacecolor='#5e3c99', markeredgecolor='black', markeredgewidth=0.5, ecolor='black', elinewidth=0.5, fmt='o', alpha=0.8)
                    ax.errorbar(x_tilings_polymer_packing, y_value_polymer_packing, label="Polymer + GPAT", markersize=sqrt(18), markerfacecolor='#e66101', markeredgecolor='black', markeredgewidth=0.5, ecolor='black', elinewidth=0.5, fmt='s', alpha=0.8)
                    # Set axes labels and limits
                    ax.set_ylim(bottom=0)
                    ax.set_xlim(left=min(x_tilings)-2, right=max(x_tilings)+2)
                    ax.yaxis.grid(True)
                    ax.xaxis.grid(True)
                    ax.grid(which='both', alpha=0.3)
                    ax.set_ylabel(counter + "\nreduction over Clang-O3")
                    ax.set_xlabel('Tiling size (all dimensions)')
                    # Save the figure and show
                    graph_path = perf_relative_outputs_dir / (benchmark + '-' + counter + '.png')
                    legend = plt.legend(ncol=2, loc='lower center', frameon=True, framealpha=1, bbox_to_anchor=(0.5, 1), columnspacing=0.8, handletextpad=0.3, handlelength=1.0)
                    frame = legend.get_frame()
                    frame.set_facecolor('white')
                    frame.set_edgecolor('black')
                    plt.savefig(graph_path, bbox_inches='tight', dpi=300)
                    plt.close(fig)
                # ---------------------------------------------------------------------------------------------


