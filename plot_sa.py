import argparse
import csv

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
MUTED = '#898781'
GRID = '#e6e5e0'
CURRENT = '#2a78d6'   # blue: the config SA is currently sitting on
BEST = '#eb6834'      # orange: best config found so far


def load_trace(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['iteration'] = int(r['iteration'])
        r['accepted'] = r['accepted'] == 'True'
        for k in ('temperature', 'candidate_cost', 'current_cost', 'best_cost'):
            r[k] = float(r[k])
    return rows


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(rows, output):
    it = [r['iteration'] for r in rows]
    # Plot accuracy (1 - cost) rather than cost: easier to read
    cand = [1 - r['candidate_cost'] for r in rows]
    current = [1 - r['current_cost'] for r in rows]
    best = [1 - r['best_cost'] for r in rows]
    temp = [r['temperature'] for r in rows]

    fig, (ax_acc, ax_temp) = plt.subplots(
        2, 1, figsize=(8, 5.5), sharex=True,
        gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.12},
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_acc, ax_temp):
        style_axis(ax)

    acc_it = [i for i, r in zip(it, rows) if r['accepted']]
    acc_y = [y for y, r in zip(cand, rows) if r['accepted']]
    rej_it = [i for i, r in zip(it, rows) if not r['accepted']]
    rej_y = [y for y, r in zip(cand, rows) if not r['accepted']]

    ax_acc.step(it, current, where='post', color=CURRENT, linewidth=2, label='current config')
    ax_acc.step(it, best, where='post', color=BEST, linewidth=2, label='best so far')
    ax_acc.scatter(acc_it, acc_y, s=40, color=CURRENT, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label='candidate (accepted)')
    ax_acc.scatter(rej_it, rej_y, s=40, facecolor='none', edgecolor=MUTED, linewidth=1.5,
                   zorder=3, label='candidate (rejected)')

    # Direct labels at the right end of each line (merged when they coincide)
    x_end = it[-1] + 0.3
    if abs(current[-1] - best[-1]) < 1e-9:
        ax_acc.text(x_end, best[-1], 'current = best', color=INK_SECONDARY, fontsize=9, va='center')
    else:
        ax_acc.text(x_end, current[-1], 'current', color=INK_SECONDARY, fontsize=9, va='center')
        ax_acc.text(x_end, best[-1], 'best', color=INK_SECONDARY, fontsize=9, va='center')

    ax_acc.set_ylabel('validation accuracy', color=INK_SECONDARY, fontsize=10)
    ax_acc.set_title('Simulated annealing search', loc='left', color=INK, fontsize=12, pad=30)
    # Legend in a row above the plot so it never covers data
    ax_acc.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, frameon=False,
                  fontsize=9, labelcolor=INK_SECONDARY, handletextpad=0.4, columnspacing=1.2)

    ax_temp.plot(it, temp, color=MUTED, linewidth=2)
    ax_temp.set_ylabel('temperature', color=INK_SECONDARY, fontsize=10)
    ax_temp.set_xlabel('SA iteration', color=INK_SECONDARY, fontsize=10)
    ax_temp.set_ylim(bottom=0)
    ax_temp.set_xlim(it[0] - 0.5, it[-1] + 2.5)

    fig.savefig(output, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    print(f"Saved plot to {output}")


def main():
    parser = argparse.ArgumentParser(description='Plot the SA trace written by train.py')
    parser.add_argument('--log', default='sa_log.csv')
    parser.add_argument('--output', default='sa_trace.png')
    args = parser.parse_args()
    plot(load_trace(args.log), args.output)


if __name__ == '__main__':
    main()
