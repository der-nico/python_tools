import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',
       r'\sisetup{detect-all}',
       r'\usepackage{helvet}',
       r'\usepackage[EULERGREEK]{sansmath}',
       r'\sansmath',
       r'\usepackage{amsmath}'
       ]  
plt.rcParams["text.latex.preview"] = True

from collections import namedtuple

class data_sample(namedtuple('sample', [
                           'index',
                           'draw_type',
                           'values',
                           'stat',
                           'total_err_up',
                           'total_err_down',
                           'name',
                           'color',
                           'draw_style',
                           'denominator'
                       ])):
    def __new__(cls,
                index=-99,
                draw_type='point',
                values=[],
                stat=[],
                total_err_up=[],
                total_err_down=[],
                name='no name',
                color='k',
                draw_style=[],
                denominator=-99):
        return super(data_sample, cls).__new__(cls, index, draw_type, values, stat,
                                               total_err_up, total_err_down,
                                               name, color, draw_style, denominator)

def make_plots_great_again(canvas, handles=None, labels=None, do_set_axis=True, do_xlim=True, do_ylim=True, do_removeoverlap=True, do_best_legendpos=True, edge_space=0.03, nbins_x=7, nbins_y=7, labelsize=15, tick_length_major=10, tick_length_minor=4, legend_loc=[-99.,-99]):
    canvas.figure.canvas.draw()
    if do_set_axis:
        set_axis(canvas, labelsize=labelsize, nbins_x=nbins_x, nbins_y=nbins_y, tick_length_major=tick_length_major, tick_length_minor=tick_length_minor)
    data_min, data_max = get_plotted_data_lim(canvas, line2D=True, lineCollection=True, PolyCollection=True, use_isfinite=True, limit_x_range=True, limit_y_range=True)
    if do_xlim:
        canvas.set_xlim(min(np.asarray(data_max)[0,0],np.asarray(data_min)[0,0]),max(np.asarray(data_max)[-1,0],np.asarray(data_min)[-1,0]))
    if do_ylim:
        canvas.set_ylim(np.min(np.asarray(data_min)[:,1]),np.max(np.asarray(data_max)[:,1]))
    if do_removeoverlap and not do_best_legendpos:
        remove_data_text_overlap(canvas, legend=True, edge_space=edge_space, limit_x_range=True, limit_y_range=True)
    if do_best_legendpos:
        if legend_loc[0] == -99 and legend_loc[1] == -99:
            best_pos_x, best_pos_y = best_legend_pos(canvas, silence=True, top=True, legend_loc=legend_loc)
            if handles and labels:
                canvas.legend(handles, labels, loc=(best_pos_x, best_pos_y), fontsize=15, frameon=False, handletextpad=0.5, numpoints=1, columnspacing=0.5)
            else:
                canvas.legend(loc=(best_pos_x, best_pos_y), fontsize=15, frameon=False, handletextpad=0.5, numpoints=1, columnspacing=0.5)
        else:
            if remove_data_text_overlap:
                remove_data_text_overlap(canvas, legend=False, edge_space=edge_space, limit_x_range=True, limit_y_range=True)
            if handles and labels:
                canvas.legend(handles, labels, loc=legend_loc, fontsize=15, frameon=False, handletextpad=0.5, numpoints=1, columnspacing=0.5)
            else:
                canvas.legend(loc=legend_loc, fontsize=15, frameon=False, handletextpad=0.5, numpoints=1, columnspacing=0.5)




    
    


def plot_hist(canvas, bins_step, sample):
    values_step = np.repeat(sample.values, 2)
    canvas.plot(bins_step, values_step, color=sample.color, linestyle='--', linewidth=2.0, label=sample.name,dashes=sample.draw_style)




def plot_points(canvas, bins_center, sample):
    err_up = (sample.total_err_up if (len(sample.total_err_up) > 0)
        else sample.total_err_down if (len(sample.total_err_down) > 0)
        else sample.stat if (len(sample.stat) > 0 )
        else [])

    err_down = (sample.total_err_down if (len(sample.total_err_down) > 0)
        else sample.total_err_up if (len(sample.total_err_up) > 0)
        else sample.stat if (len(sample.stat) > 0)
        else [])
    if sample.draw_style=='enterprise':
        canvas.errorbar(bins_center, sample.values, yerr=[err_up, err_down], ls='none', marker=enterprise, markersize=20, markeredgewidth=0.2, color=sample.color, capsize=0, label=sample.name)
    elif sample.draw_style=='enterprise_cc':
        canvas.errorbar(bins_center, sample.values, yerr=[err_up, err_down], ls='none', marker=enterprise_cc, markersize=30, markeredgewidth=0.2, color=sample.color, capsize=0, label=sample.name)
    else:
        canvas.errorbar(bins_center, sample.values, yerr=[err_up, err_down], fmt=sample.draw_style, color=sample.color, capsize=0, label=sample.name)



def transform_cartesian_to_polar(circle_origin, cord_0, cord_1, inverse=False):
    if not inverse:
        cord_0_new = ((circle_origin[0]-cord_0)**2 + (circle_origin[1]-cord_1)**2)**0.5
        cord_1_new = math.atan2((circle_origin[0]-cord_0), (circle_origin[1]-cord_1))
    else:
        cord_0_new = cord_0 * np.cos(cord_1) + circle_origin[0]
        cord_1_new = cord_0 * np.sin(cord_1) + circle_origin[1]
    return cord_0_new,cord_1_new


def get_limits(x_vals, y_mins, y_maxs, spaces_upper, spaces_lower, log=False, lim_before=(-99, -99)):
    edge_space = 0.03 
    y_max_global = np.nanmax(y_maxs)
    y_min_global = np.nanmin(y_mins)
    upper_warning=False
    lower_warning=False
    # print('before', y_min_global,y_max_global)
    if log:
        if lim_before[0] != -99:
            # print('bef',y_min_global)
            y_min_global = np.nanmin(np.where(y_mins>lim_before[0], y_mins, lim_before[0]))
            # print('aft',y_min_global)

            # print('Warning: not all lower limits are inside the plot')
            lower_warning = True
        if lim_before[1] != -99:
            y_max_global = np.nanmax(np.where(y_maxs>lim_before[1], y_maxs, lim_before[1]))
            # print('Warning: not all upper limits are inside the plot')
            upper_warning = True
        y_max_global = np.log10(y_max_global)
        y_min_global = np.log10(y_min_global)
        # print('111',y_min_global)

    # print('diff',y_diff_global)
    y_exclude_max = 1
    y_exclude_min = 0
    y_val_min = y_min_global
    y_val_max = y_max_global
    y_min_global, y_max_global = add_space(y_min_global, y_max_global, space_upper= 0 if upper_warning else edge_space, space_lower=0 if lower_warning else edge_space, log=False)
    
    # print('222',y_min_global)
    y_diff_global = max(y_max_global -y_min_global, 10e-99)
    changed = True  
    while changed:
        changed = False
        biggest_need_lower = 0
        biggest_need_upper = 0

        for x_val, y_min, y_max, space_upper, space_lower in zip(x_vals, y_mins, y_maxs, spaces_upper, spaces_lower):
            if log:
                if lim_before[0] != -99:
                    lim_low = lim_before[0]
                else:
                    lim_low = -10e10
                if lim_before[1] != -99:
                    lim_up = lim_before[1]
                else:
                    lim_up = 10e10
                y_min =np.log10(min(max(y_min,lim_low),lim_up))
                y_max =np.log10(min(max(y_max,lim_low),lim_up))
            existing_space_upper = ((y_max_global - y_max) / y_diff_global)
            existing_space_lower = ((y_min-y_min_global) / y_diff_global)
            # print('here',space_lower, space_upper, existing_space_lower, existing_space_upper)
            if space_upper > existing_space_upper:
                # print('and here',space_upper -existing_space_upper,biggest_need_upper)
                if biggest_need_upper < space_upper+edge_space - existing_space_upper:
                    # print("rete", y_exclude_max , space_upper)
                    biggest_need_upper = space_upper+edge_space - existing_space_upper
                    y_exclude_max = 1 - space_upper
                    y_val_max = y_max
                    changed=True
            if space_lower > existing_space_lower :
                # print('or here',space_lower -existing_space_lower,biggest_need_lower)
                if biggest_need_lower < space_lower + edge_space - existing_space_lower:
                    biggest_need_lower = space_lower + edge_space - existing_space_lower
                    # print("ratata", y_exclude_min, space_lower)
                    y_exclude_min = space_lower
                    y_val_min = y_min
                    changed=True
        # print('rheyheh',1 - y_exclude_max +  edge_space, y_exclude_min + edge_space)
        space_upper = 1 - y_exclude_max + edge_space
        if upper_warning and y_val_max == lim_before[1] and y_exclude_max == 1:
            space_upper = 0
        space_lower = y_exclude_min + edge_space
        if lower_warning and y_val_min == lim_before[0] and y_exclude_min == 0:
            space_lower = 0

        y_min_global, y_max_global = add_space(y_val_min, y_val_max, space_upper=space_upper, space_lower=space_lower, log=False)
        # print('333',y_min_global)
        y_diff_global = y_max_global -y_min_global 
        # print('after1', y_min_global,y_max_global)

    # print('test', y_exclude_min + edge_space, 1 - y_exclude_max + edge_space, y_val_min,y_val_max,y_max_global-y_min_global)
    space_upper = 1 - y_exclude_max + edge_space
    if upper_warning and y_val_max == lim_before[1] and y_exclude_max == 1:
        space_upper = 0
    space_lower = y_exclude_min + edge_space
    if lower_warning and y_val_min == lim_before[0] and y_exclude_min == 0:
        space_lower = 0
    y_min_global, y_max_global = add_space(y_val_min, y_val_max, space_upper=space_upper, space_lower=space_lower, log=False)
    if log:
        y_min_global = 10**(y_min_global)
        y_max_global = 10**(y_max_global)
    # print('after', y_min_global,y_max_global)
    return y_min_global, y_max_global

def add_space(lim_lower, lim_upper, space_upper=0., space_lower=0., log=False):
    if space_lower +space_upper >= 1:
        print("ERROR Requested space is larger 1 initial values are returned")
        return lim_lower, lim_upper
        
    if log:
        lim_upper = np.log10(lim_upper)
        lim_lower = np.log10(lim_lower)

    lim_diff = (lim_upper - lim_lower) / (1 - space_upper - space_lower)
    lim_upper += lim_diff * space_upper
    lim_lower -= lim_diff * space_lower
    if log:
        lim_lower = 10**(lim_lower)
        lim_upper = 10**(lim_upper)
    return lim_lower, lim_upper

# major_low_yx
def get_priorities(x_axis,y_axis, x_tick=None, y_tick=None, order='minor_xy_high'):
    order = order.split('_')
    major_base, minor_base = 0, 0
    low_base, high_base = 0, 0
    x_base, y_base = 0, 0
    x_base_additional, y_base_additional = 0, 0
    for i in range(len(order)):
        if order[i] == 'major':
            major_base = 10**(len(order)-i)
        if order[i] == 'minor':
            minor_base = 10**(len(order)-i)
        if order[i] == 'low':
            low_base = 10**(len(order)-i)
        if order[i] == 'high':
            high_base = 10**(len(order)-i)
        if order[i] == 'yx':
            y_base_additional = 1
            y_base = 10**(len(order)-i)
        if order[i] == 'xy':
            x_base_additional = 1
            x_base = 10**(len(order)-i)
    y_priority_temp = y_base
    x_priority_temp = x_base
    if "major" in y_axis.split("_"):
        y_priority_temp += major_base + y_base_additional
    else:
        y_priority_temp += minor_base + y_base_additional
    if "up" in x_axis.split("_"):
        y_priority_temp += low_base + y_base_additional
        if y_tick and not y_tick.tick1On:
            y_priority_temp = -1
    else: 
        y_priority_temp += high_base + y_base_additional
        if y_tick and not y_tick.tick2On:
            y_priority_temp = -1

    if "major" in x_axis.split("_"):
        x_priority_temp += major_base + x_base_additional
    else:
        x_priority_temp += minor_base + x_base_additional
    if "up" in y_axis.split("_"):
        x_priority_temp += low_base + x_base_additional
        if x_tick and not x_tick.tick1On:
            x_priority_temp = -1
    else: 
        x_priority_temp += high_base + x_base_additional
        if x_tick and not x_tick.tick2On:
            x_priority_temp = -1


    return x_priority_temp, y_priority_temp

def get_plotted_data_lim(canvas, line2D=True, lineCollection=True, PolyCollection=True, use_isfinite=True, limit_x_range=True, limit_y_range=False):
    if limit_x_range:
        xlim = canvas.get_xlim()
    else:
        xlim = (-10e99, 10e99)
    if limit_y_range:
        ylim = canvas.get_ylim()
    else:
        ylim = (-10e99, 10e99)
    yscale = canvas.get_yscale()
    if 'log' in yscale:
        log_y = True
    else:
        log_y = False

    data = []
    children = canvas.get_children()
    for child in children:
        if line2D and type(child) == matplotlib.lines.Line2D:
            data.extend([(x,y) for (x, y) in zip(child.get_xdata(), child.get_ydata())])
            data.extend([(x,y) for (x, y) in zip(child.get_xdata()[1:], child.get_ydata()[:-1])])
            data.extend([(x,y) for (x, y) in zip(child.get_xdata()[-1:], child.get_ydata()[1:])])
        if lineCollection and type(child) == matplotlib.collections.LineCollection:
            data.extend([point for line in child.get_segments() for point in line])
            data.extend([point for path in child.get_paths() for point in path.get_extents().get_points()])
        if PolyCollection and type(child) == matplotlib.collections.PolyCollection:
            data.extend([point for path in child.get_paths() for point in path.vertices])
    data = sorted(data, key = lambda x: x[0])
    data_max = []
    data_min = []
    this_x = None
    y_list = []
    for x,y in data:
        if x == this_x:
            if np.isfinite(y) and use_isfinite and ylim[0] <= y and y <= ylim[1]:
                y_list.append(y)
        else:
            if len(y_list) > 0:
                if limit_x_range and xlim[0] <= this_x <= xlim[1]:
                    data_max.append((this_x,np.nanmax(y_list)))
                    data_min.append((this_x,np.nanmin(y_list)))
            this_x = x
            y_list = []
            if np.isfinite(y) and use_isfinite and ylim[0] <= y and y <= ylim[1]:
                y_list.append(y)
    if len(y_list) > 0:
        # if not log_y:
        if limit_x_range and xlim[0] <= this_x <= xlim[1]:
            data_max.append((this_x,np.nanmax(y_list)))
            data_min.append((this_x,np.nanmin(y_list)))
    return data_min, data_max


def do_overlap(box0, box1):
    box0_x0, box0_y0, box0_x1, box0_y1 = box0.get_window_extent().get_points().flatten()
    box1_x0, box1_y0, box1_x1, box1_y1 = box1.get_window_extent().get_points().flatten()
    if (box0_x0 > box1_x1 or box1_x0 > box0_x1):
        return False
    if (box0_y0 > box1_y1 or box1_y0 > box0_y1):
        return False
    return True


def get_text_boxes(canvas, text=True, legend=False):
    text_box = []
    children = canvas.get_children()
    for child in children:
        if legend and type(child) == matplotlib.legend.Legend:
            if do_overlap(child, canvas):
                text_box.append(child.get_window_extent().get_points())
        if text and type(child) == matplotlib.text.Text:
            if do_overlap(child, canvas):
                text_box.append(child.get_window_extent().get_points())
    return text_box

def remove_data_text_overlap(canvas, legend=False, edge_space = 0.03, limit_x_range=True, limit_y_range=False):  
    canvas.figure.canvas.draw()
    canvas_extend = canvas.get_window_extent().get_points()
    # legend_extend = canvas.legend_.get_window_extent().get_points()
    excludes = get_text_boxes(canvas, text=True, legend=legend)
    data_min, data_max = get_plotted_data_lim(canvas, line2D=True, lineCollection=True, limit_x_range=limit_x_range, limit_y_range=limit_y_range)
    lower_warning = False
    upper_warning = False

    if len(data_max)>0:
        yscale = canvas.get_yscale()
        xscale = canvas.get_xscale()
        if 'log' in yscale:
            log_y = True
            lim_before = canvas.get_ylim()
        else:
            log_y = False
            lim_before = (-99, -99)
        if 'log' in xscale:
            log_x = True
        else:
            log_x = False
        y_lim_down, y_lim_up = canvas.get_ylim()
        x_vals = []
        y_mins = []
        y_maxs = []
        spaces_upper = []
        spaces_lower = []

        for (x_min, val_min), (x_max, val_max) in zip(data_min, data_max):
            if log_y:
                if lim_before[0]>val_min:
                    lower_warning = True
                if lim_before[1]<val_max:
                    uppper_warning = True
            # print("max",val_max)
            space_upper = 0
            space_lower = 0
            for exclude in excludes:
                exclude_x_min, _ = axis_data_coords_sys_transform(canvas, (exclude[0][0]-canvas_extend[0][0])/(canvas_extend[1][0] - canvas_extend[0][0]) - edge_space, 0)
                exclude_y_min = (exclude[0][1]-canvas_extend[0][1])/(canvas_extend[1][1] - canvas_extend[0][1])
                exclude_x_max, _ = axis_data_coords_sys_transform(canvas, (exclude[1][0]-canvas_extend[0][0])/(canvas_extend[1][0] - canvas_extend[0][0]) + edge_space, 0)
                exclude_y_max = (exclude[1][1]-canvas_extend[0][1])/(canvas_extend[1][1] - canvas_extend[0][1])
                if (0<exclude_y_min<1) or (0<exclude_y_max<1):
                    if (exclude_x_min < x_min < exclude_x_max) or (exclude_x_min < x_max < exclude_x_max):
                        if exclude_y_min > 0.5:
                            space_upper = max(space_upper, 1 - exclude_y_min)
                        if exclude_y_min < 0.5:
                            space_lower = max(space_lower, exclude_y_max)
            x_vals.append(x_min)
            y_mins.append(val_min)
            y_maxs.append(val_max)
            spaces_upper.append(space_upper)
            spaces_lower.append(space_lower)

        # y_lim_down, y_lim_up = get_limits(x_vals=x_vals, y_mins=y_mins, y_maxs=y_maxs, spaces_upper=spaces_upper, spaces_lower=spaces_lower, log=log_y)
        
        if not lower_warning: 
            lim_before = (-99,lim_before[1])
        if not upper_warning: 
            lim_before = (lim_before[0], -99)
        y_lim_down, y_lim_up = get_limits(x_vals=x_vals, y_mins=y_mins, y_maxs=y_maxs, spaces_upper=spaces_upper, spaces_lower=spaces_lower, log=log_y, lim_before=lim_before)
        canvas.set_ylim(y_lim_down,y_lim_up)
    else:
        print('ERROR no data found, the plot was not changed')


def best_legend_pos(canvas, legend_loc=[-99.,-99], top=False, bottom=False, left=False, right=False, edge_space=0.03, y_min=None, silence=False):
    if legend_loc[0] != -99:
        if left or right:
            print('Error legend x position and x position preference are given given legend_loc is used')
    if legend_loc[1] != -99:
        if legend_loc[1]>0.5:
            top = True if not bottom else top
        if legend_loc[1]<0.5:
            bottom = True if not top else bottom

    if top and bottom:
        print('Error legend can\'tbe at the top and the bottom (top is used)')
        bottom = False

    if right and left:
        print('Error legend can\'tbe at the right and the left (right is used)')
        left = False
    if not silence:
        print('NOTICE: The limits of the plot are changed with this function (best_legend_pos)')
    canvas.figure.canvas.draw()
    canvas_extend = canvas.get_window_extent().get_points()
    print(canvas_extend)
    legend_extend = canvas.legend_.get_window_extent().get_points()
    legend_width = (legend_extend[1][0] - legend_extend[0][0]) / (canvas_extend[1][0] - canvas_extend[0][0])
    legend_heigth = (legend_extend[1][1] - legend_extend[0][1]) / (canvas_extend[1][1] - canvas_extend[0][1])
    print(legend_extend)
    excludes = get_text_boxes(canvas, text=True, legend=False)
    print(excludes)
    data_min, data_max = get_plotted_data_lim(canvas, line2D=True, lineCollection=True, limit_x_range=True, limit_y_range=True)
    print(data_min, data_max)
    # xscale = canvas.get_xscale()
    yscale = canvas.get_yscale()
    lower_warning = False
    upper_warning = False
    if 'log' in yscale:
        log_y = True
        lim_before = canvas.get_ylim()
    else:
        log_y = False
        lim_before = (-99, -99)
    if y_min:
        lim_before[0] = y_min
    best_pos_x = -99
    best_pos_y = -99
    best_y_space = 0
    best_data_edge_y = -99
    y_exclude_max_global = 1
    y_exclude_min_global = 0
    # print(canvas.get_ylim())
    remove_data_text_overlap(canvas, edge_space=edge_space, limit_y_range=True)
    y_lim_down_best, y_lim_up_best = canvas.get_ylim()
    # print(y_lim_down_best, y_lim_up_best)
    # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    y_lim_diff = 10e100
    leg_top = True
    possible_x_positions = np.linspace(edge_space, 1 - edge_space - legend_width,10)
    possible_x_positions = [possible_x_positions[-i//2-1] if i%2==0 else possible_x_positions[i//2] for i in range(len(possible_x_positions))]
    if left:
        possible_x_positions =[edge_space]
    if right:
        possible_x_positions =[1 - edge_space - legend_width]
    if legend_loc[0] != -99:
        possible_x_positions =[legend_loc[0]]

    for leg_pos_x in possible_x_positions:
        skip_bottom = False
        skip_top    = False

        legend_pos_2 = edge_space
        legend_pos_1 = 1 - (legend_heigth+edge_space)

        leg_edge_x_min, _ = axis_data_coords_sys_transform(canvas, leg_pos_x-edge_space, 0)
        leg_edge_x_max, _ = axis_data_coords_sys_transform(canvas, leg_pos_x+legend_width+edge_space, 0)
        for exclude in excludes:
            exclude_x_min, _ = axis_data_coords_sys_transform(canvas, (exclude[0][0]-canvas_extend[0][0])/(canvas_extend[1][0] - canvas_extend[0][0]), 0)
            exclude_y_min = (exclude[0][1]-canvas_extend[0][1])/(canvas_extend[1][1] - canvas_extend[0][1])
            exclude_x_max, _ = axis_data_coords_sys_transform(canvas, (exclude[1][0]-canvas_extend[0][0])/(canvas_extend[1][0] - canvas_extend[0][0]), 0)
            exclude_y_max = (exclude[1][1]-canvas_extend[0][1])/(canvas_extend[1][1] - canvas_extend[0][1])

            if (leg_edge_x_min < exclude_x_min < leg_edge_x_max) or (leg_edge_x_min< exclude_x_max < leg_edge_x_max):
                if exclude_y_min > 0.5:
                    if legend_pos_1 > exclude_y_min-legend_heigth-edge_space:
                        legend_pos_1 = exclude_y_min-legend_heigth-edge_space
                if exclude_y_min < 0.5:
                    if legend_pos_2 < (exclude_y_max + edge_space):
                        legend_pos_2 = exclude_y_max + edge_space

        data_edge_y_max = -99
        data_edge_y_min = 1000000
        x_vals = []
        y_mins = []
        y_maxs = []
        spaces_upper = []
        spaces_lower = []
        spaces_upper2 = []
        spaces_lower2 = []
        for (x_min, val_min), (x_max, val_max) in zip(data_min, data_max):
            if log_y:
                if lim_before[0]>val_min:
                    lower_warning = True
                if lim_before[1]<val_max:
                    uppper_warning = True
            space_upper = 0
            space_lower = 0
            space_upper2 = 0
            space_lower2 = 0
            for exclude in excludes:
                exclude_x_min, _ = axis_data_coords_sys_transform(canvas, (exclude[0][0]-canvas_extend[0][0])/(canvas_extend[1][0] - canvas_extend[0][0]) - edge_space, 0)
                exclude_y_min = (exclude[0][1]-canvas_extend[0][1])/(canvas_extend[1][1] - canvas_extend[0][1])
                exclude_x_max, _ = axis_data_coords_sys_transform(canvas, (exclude[1][0]-canvas_extend[0][0])/(canvas_extend[1][0] - canvas_extend[0][0]) + edge_space, 0)
                exclude_y_max = (exclude[1][1]-canvas_extend[0][1])/(canvas_extend[1][1] - canvas_extend[0][1])
                print(exclude_x_min, exclude_x_max, exclude_y_min, exclude_y_max)
                if (0<exclude_y_min<1) or (0<exclude_y_max<1):
                    if (exclude_x_min  < x_min < exclude_x_max) or (exclude_x_min < x_max < exclude_x_max):
                        if exclude_y_min > 0.5:
                            # print(space_upper, space_upper2, 1-exclude_y_min)
                            space_upper = max(space_upper, 1 - exclude_y_min)
                            space_upper2 = max(space_upper2, 1 - exclude_y_min)
                        if exclude_y_min < 0.5:
                            space_lower = max(space_lower, exclude_y_max)
                            space_lower2 = max(space_lower2, exclude_y_max)
            if (leg_edge_x_min < x_min < leg_edge_x_max) or (leg_edge_x_min < x_max < leg_edge_x_max):
                if log_y:
                    if lim_before[0]>val_min:
                        skip_bottom = True
                    if lim_before[1]<val_max:
                        skip_top = True
                # print('bexcludes',space_upper,space_lower,space_upper2,space_lower2)
                space_upper = max(space_upper, 1 - legend_pos_1 + edge_space)
                space_lower2 = max(space_lower2, legend_pos_2 + legend_heigth + edge_space)

                                # print('aexcludes',space_upper,space_lower,space_upper2,space_lower2, legend_pos_1, legend_pos_2)

            x_vals.append(x_min)
            y_mins.append(val_min)
            y_maxs.append(val_max)
            spaces_upper.append(space_upper)
            spaces_lower.append(space_lower)
            spaces_upper2.append(space_upper2)
            spaces_lower2.append(space_lower2)
        # print(spaces_upper, spaces_upper2)


        if not skip_top and not bottom and len(y_maxs)>0:
            # if abs(leg_pos_x -0.3662850387455) < 0.0001:
            #     y_lim_down_new, y_lim_up_new = get_limits(x_vals=x_vals, y_mins=y_mins, y_maxs=y_maxs, spaces_upper=spaces_upper, spaces_lower=spaces_lower, log=log_y)
            # else:
            if not lower_warning: 
                lim_before = (-99,lim_before[1])
            if not upper_warning: 
                lim_before = (lim_before[0], -99)

            y_lim_down_new, y_lim_up_new = get_limits(x_vals=x_vals, y_mins=y_mins, y_maxs=y_maxs, spaces_upper=spaces_upper, spaces_lower=spaces_lower, log=log_y, lim_before=lim_before)

            # y_space = y_exclude_max - np.abs(axis_data_coords_sys_transform(canvas, 0, data_edge_y_max, inverse=True)[1])
            # print('diffdiff',y_lim_diff)
            # print(y_lim_up_new-y_lim_down_new)
            if (y_lim_up_new-y_lim_down_new < y_lim_diff) or (y_lim_up_new-y_lim_down_new == y_lim_diff and not leg_top):
                # print("new_top",y_lim_down_new, y_lim_up_new)
                y_lim_diff = y_lim_up_new-y_lim_down_new
                y_lim_down_best = y_lim_down_new
                y_lim_up_best = y_lim_up_new
                best_pos_x = leg_pos_x
                best_pos_y = legend_pos_1
                leg_top = True
        if not skip_bottom and not top and len(y_maxs)>0:
            if not lower_warning: 
                lim_before = (-99,lim_before[1])
            if not upper_warning: 
                lim_before = (lim_before[0], -99)
            y_lim_down_new, y_lim_up_new = get_limits(x_vals=x_vals, y_mins=y_mins, y_maxs=y_maxs, spaces_upper=spaces_upper2, spaces_lower=spaces_lower2, log=log_y, lim_before=lim_before)
            # print('diffdiff',y_lim_diff)
            # print(y_lim_up_new-y_lim_down_new)
            if y_lim_up_new-y_lim_down_new < y_lim_diff:
                # print("new_bottom",y_lim_down_new, y_lim_up_new)
                y_lim_diff = y_lim_up_new-y_lim_down_new
                y_lim_down_best = y_lim_down_new
                y_lim_up_best = y_lim_up_new
                best_pos_x = leg_pos_x
                best_pos_y = legend_pos_2
                leg_top = False

    # y_lim_down, y_lim_up = canvas.get_ylim()
    # y_lim_1 = y_lim_down, y_lim_up
    # y_lim_1 = add_space(y_lim_down, y_lim_up, space_upper=1-y_exclude_max_global, space_lower=y_exclude_min_global, log=log_y)
    # if legend_loc[0] != -99:
    #     y_lim_2 = add_space(y_lim_down, best_data_edge_y, space_upper= 1 - legend_loc[1]-edge_space, space_lower=y_exclude_min_global, log=log_y)
    #     y_lim = (y_lim_1[0], max(y_lim_1[1], y_lim_2[1]))
    # else:
    #     if leg_top:
    #         y_lim_2 = add_space(y_lim_down, best_data_edge_y, space_upper=1-best_pos_y+edge_space, space_lower=y_exclude_min_global, log=log_y)
    #         y_lim = (y_lim_1[0], max(y_lim_1[1], y_lim_2[1]))
    #     else:
    #         y_lim_2 = add_space(best_data_edge_y, y_lim_up, space_upper=1-y_exclude_max_global, space_lower=best_pos_y+legend_heigth+edge_space, log=log_y)
    #         y_lim = y_lim_1 if y_lim_1[0]< y_lim_2[0] else y_lim_2

    canvas.set_ylim(y_lim_down_best, y_lim_up_best)
    # print(best_pos_x, best_pos_y)
    return best_pos_x, best_pos_y


def get_ticks_and_indices(canvas, axis_name, tick_length, tick_length_up):
    if "x" in axis_name.split("_"):
        axis = canvas.xaxis
        if "major" in axis_name.split("_"):
            ticks = canvas.xaxis.get_major_ticks()
        elif "minor" in axis_name.split("_"):
            ticks = canvas.xaxis.get_minor_ticks()
        else:
            print('ERROR1')
    elif "y" in axis_name.split("_"):
        axis = canvas.yaxis
        if "major" in axis_name.split("_"):
            ticks = canvas.yaxis.get_major_ticks()
        elif "minor" in axis_name.split("_"):
            ticks = canvas.yaxis.get_minor_ticks()
        else:
            print('ERROR2')
    else:
        print('ERROR3')
    if "up" in axis_name.split("_"):
        i = 0
        done = False
        tick_indices = []
        while not done and len(ticks)>i:
            if axis.get_view_interval()[0] <= ticks[i].get_loc() <= axis.get_view_interval()[1]:
                if ticks[i].get_loc()<=tick_length:
                    tick_indices.append(i)
                # else:
                #     done = True
            i += 1
    if "down" in axis_name.split("_"):
        i = -1
        done = False
        tick_indices = []
        while not done and len(ticks)+i>0:
            if axis.get_view_interval()[1] >= ticks[i].get_loc() >= axis.get_view_interval()[0]:
                    tick_indices.append(i)
                # else:
                #     done = True
            i -= 1
    return ticks, tick_indices

def set_axis(canvas, labelsize=18, nbins_x=7, nbins_y=7, tick_length_major=10, tick_length_minor=4):
    xscale = canvas.get_xscale()
    yscale = canvas.get_yscale()
    
    if 'log' in xscale:
        log_x = True
    else:
        log_x = False

    if 'log' in yscale:
        log_y = True
    else:
        log_y = False

    canvas.tick_params(direction='in')    
    canvas.minorticks_on()
    canvas.tick_params(axis='x', labelsize=18)
    canvas.tick_params(axis='y', labelsize=18)
    canvas.tick_params(axis='x', which='minor', labelsize=18)
    canvas.tick_params(axis='y', which='minor', labelsize=18)
    # if not log_x:
    #     canvas.locator_params(axis='x', nbins=7)
    if not log_y:
        canvas.locator_params(axis='y', nbins=7)
    canvas.tick_params(length=10)
    canvas.tick_params(direction='in', which='minor', length=4)
    canvas.yaxis.set_ticks_position('both')
    canvas.xaxis.set_ticks_position('both')

def draw_circle_axis(canvas, cx, cy, ri, ro, ticks=[0, 1/4.*math.pi, math.pi/2., 3/4.*math.pi, math.pi, 5/4.*math.pi, math.pi*3/2., 7/4.*math.pi, 2*math.pi], labelsize=15):
    circle1 = plt.Circle((cx, cy), ri, color='k', fill=False)
    circle2 = plt.Circle((cx, cy), ro, color='k', fill=False)
    canvas.add_artist(circle1)
    canvas.add_artist(circle2)
    for tick in ticks:
        canvas.annotate(''.format(tick/math.pi), xy=transform_cartesian_to_polar((cx, cy), ri, -tick+math.pi, inverse=True), xytext=transform_cartesian_to_polar((cx, cy), ri-50, -tick+math.pi, inverse=True), arrowprops=dict(facecolor='black', arrowstyle="-"))
        canvas.annotate(''.format(tick/math.pi), xy=transform_cartesian_to_polar((cx, cy), 0, -tick+math.pi, inverse=True), xytext=transform_cartesian_to_polar((cx, cy), ri-100, -tick+math.pi, inverse=True), arrowprops=dict(facecolor='black', arrowstyle="-", ls='dashed'), clip_on=False)
        canvas.annotate(r'{}$\pi$'.format(tick/math.pi), xy=transform_cartesian_to_polar((cx, cy), ri, -tick+math.pi, inverse=True), xytext=transform_cartesian_to_polar((cx, cy), ri-75, -tick+math.pi, inverse=True),fontsize=labelsize)
        

def remove_axis_overlap(canvas):
    plt.draw()
    frame_points = canvas.get_window_extent().get_points()
    tick_length_x = canvas.xaxis.get_major_ticks()[0]._size if len(canvas.xaxis.get_major_ticks())>0 else 0
    tick_length_y = canvas.yaxis.get_major_ticks()[0]._size if len(canvas.yaxis.get_major_ticks())>0 else 0
    tick_length_minor_x = canvas.xaxis.get_minor_ticks()[0]._size if len(canvas.xaxis.get_minor_ticks())>0 else 0
    tick_length_minor_y = canvas.yaxis.get_minor_ticks()[0]._size if len(canvas.yaxis.get_minor_ticks())>0 else 0
    # tick_length_data_x, tick_length_data_y = axis_data_coords_sys_transform(canvas, tick_length_y/ (canvas.figure.dpi/72.), tick_length_x /(canvas.figure.dpi/72.)) 
    tick_length_data_x, tick_length_data_y = axis_data_coords_sys_transform(canvas, tick_length_y/ (frame_points[1][0] - frame_points[0][0]) * (canvas.figure.dpi/72.), tick_length_x/ (frame_points[1][1] - frame_points[0][1]) * (canvas.figure.dpi/72.)) 
    # tick_length_data_x_up, tick_length_data_y_up = axis_data_coords_sys_transform(canvas, 1-(tick_length_y)/ (canvas.figure.dpi/72.), 1-(tick_length_x)/ (canvas.figure.dpi/72.))
    tick_length_data_x_up, tick_length_data_y_up = axis_data_coords_sys_transform(canvas, 1-(tick_length_y/ (frame_points[1][0] - frame_points[0][0]) * (canvas.figure.dpi/72.)), 1-(tick_length_x/ (frame_points[1][1] - frame_points[0][1]) * (canvas.figure.dpi/72.))) 

    x_tick_length = tick_length_y / (frame_points[1][0] - frame_points[0][0])* (canvas.figure.dpi/72.)
    y_tick_length = tick_length_x / (frame_points[1][1] - frame_points[0][1])* (canvas.figure.dpi/72.)
    x_tick_length_minor = tick_length_minor_y / (frame_points[1][0] - frame_points[0][0])* (canvas.figure.dpi/72.)
    y_tick_length_minor = tick_length_minor_x / (frame_points[1][1] - frame_points[0][1])* (canvas.figure.dpi/72.)
    y_axes = ["y_major_up", "y_major_down", "y_minor_up", "y_minor_down"] 
    x_axes = ["x_major_up", "x_major_down", "x_minor_up", "x_minor_down"] 
    axes_names = sorted([(x_axis, y_axis) for x_axis in x_axes for y_axis in y_axes], key = lambda x: -1*sum(get_priorities(x[0], x[1])))
    
    
    
    all_ticks = [
                [canvas.xaxis, canvas.xaxis.get_major_ticks()],
                [canvas.xaxis, canvas.xaxis.get_minor_ticks()],
                [canvas.yaxis, canvas.yaxis.get_major_ticks()],
                [canvas.yaxis, canvas.yaxis.get_minor_ticks()]
                ]
    for ticks in all_ticks:
        for tick_index in range(len(ticks[1])):
            if ticks[1][tick_index].get_loc()<=ticks[0].get_view_interval()[0] or ticks[1][tick_index].get_loc()>=ticks[0].get_view_interval()[1]:
                ticks[1][tick_index].tick1On = False
                ticks[1][tick_index].tick2On = False

    for (x_axis, y_axis) in axes_names:
        y_ticks, y_indices = get_ticks_and_indices(canvas, y_axis, tick_length_data_y, tick_length_data_y_up)# x_tick_length)
        x_ticks, x_indices = get_ticks_and_indices(canvas, x_axis, tick_length_data_x, tick_length_data_x_up)#, y_tick_length)
        if len(y_indices) > 0 and len(x_indices)>0:
            for y_index in y_indices:
                for x_index in x_indices:

                    y_min_x = 0
                    y_max_x = x_tick_length if "major" in y_axis.split("_") else x_tick_length_minor
                    if "down" in x_axis.split("_"):
                        y_min_x = 1-y_min_x
                        y_max_x = 1-y_max_x
                    x_pos, y_pos = axis_data_coords_sys_transform(canvas, x_ticks[x_index].get_loc(), y_ticks[y_index].get_loc(), inverse=True) 
                    x_min_y = 0
                    x_max_y = y_tick_length if "major" in x_axis.split("_") else y_tick_length_minor
                    if "down" in y_axis.split("_"):
                        x_min_y = 1-x_min_y
                        x_max_y = 1-x_max_y
                    if (min(y_min_x, y_max_x) < x_pos < max(y_min_x, y_max_x)) and (min(x_min_y, x_max_y) < y_pos < max(x_min_y, x_max_y)):
                        x_priority, y_priority = get_priorities(x_axis, y_axis, x_ticks[x_index], y_ticks[y_index])
                        if y_priority < x_priority:
                            if "up" in x_axis.split("_") :
                                y_ticks[y_index].tick1On = False
                            if "down" in x_axis.split("_") :
                                y_ticks[y_index].tick2On = False
                        else:
                            if "up" in y_axis.split("_") :
                                x_ticks[x_index].tick1On = False
                            if "down" in y_axis.split("_") :
                                x_ticks[x_index].tick2On = False
 
def axis_data_coords_sys_transform(axis_obj_in,xin,yin,inverse=False):
    """ inverse = False : Axis => Data
                = True  : Data => Axis
    """
    xscale = axis_obj_in.get_xscale()
    yscale = axis_obj_in.get_yscale()

    xlim = axis_obj_in.get_xlim()
    ylim = axis_obj_in.get_ylim()

    if 'log' in xscale:
        xlim = np.log10(xlim)
    if 'log' in yscale:
        ylim = np.log10(ylim)
    
    xdelta = xlim[1] - xlim[0]
    ydelta = ylim[1] - ylim[0]

    if not inverse:
        xout =  xlim[0] + xin * xdelta
        yout =  ylim[0] + yin * ydelta
        if 'log' in xscale:
            xout = 10**(xout)
        if 'log' in yscale:
            yout = 10**(yout)
    else:
        if 'log' in xscale:
            xin = np.log10(xin)
        if 'log' in yscale:
            yin = np.log10(yin)
        xdelta2 = xin - xlim[0]
        ydelta2 = yin - ylim[0]
        xout = xdelta2 / xdelta
        yout = ydelta2 / ydelta
    return xout,yout




def rotate_marker(marker, rot_angle):
    offset_x = 0
    offset_y = 0
    rotated_marker = []
    for point in marker:
        rotated_marker.append(((point[0] - offset_x) * math.cos(rot_angle)
            - (point[1] - offset_y) * math.sin(rot_angle),
            (point[0] - offset_x) * math.sin(rot_angle)
            + (point[1] - offset_y) * math.cos(rot_angle)) )
    return rotated_marker

marker_path = '/afs/cern.ch/user/n/nscharmb/private/python/MARKERS/' 
enterprise_cc = np.load(marker_path + 'enterprise_circle_centered.npy')
enterprise = [
        (9.08, 0),          (9.072, 0.29),      (9.064, 0.457),
        (9.052, 0.582),     (9.02, 0.829),      (8.984, 1.072), 
        (8.942, 1.27),      (8.906, 1.423),     (8.824, 1.725),
        (8.74, 1.961),      (8.692, 2.088),     (8.646, 2.201),
        (8.542, 2.429),     (8.422, 2.654),     (8.262, 2.915),
        (8.118, 3.116),     (7.984, 3.298),     (7.872, 3.426),
        (7.68, 3.632),      (7.504, 3.798),     (7.238, 4.019),
        (7.024, 4.171),     (6.808, 4.318),     (6.544, 4.459),
        (6.354, 4.553),     (6.152, 4.637),     (5.872, 4.739),
        (5.55, 4.83),       (5.34, 4.879),      (5.144, 4.911),
        (4.988, 4.928),     (4.708, 4.955),     (4.498, 4.961),
        (4.286, 4.96),      (4.11, 4.95),       (3.814, 4.919),
        (3.508, 4.869),     (3.292, 4.814),     (3.112, 4.763),
        (2.836, 4.667),     (2.678, 4.601),     (2.516, 4.524),
        (2.248, 4.382),     (2.092, 4.287),     (1.912, 4.167),
        (1.706, 4.01),      (1.508, 3.841),     (1.37, 3.71),
        (1.24, 3.572),      (1.142, 3.468),     (1.08, 3.391),
        (0.984, 3.273),     (0.904, 3.165),     (0.696, 2.856),
        (0.48, 2.459),      (0.39, 2.268),      (0.348, 2.168),
        (0.302, 2.05),      (0.234, 1.862),     (0.168, 1.679),
        (0.11, 1.472),      (0.06, 1.287),      (0.03, 1.158),
        (0.022, 1.056),     (-2.12, 0.973),     (-3.832, 2.92),
        (-1.336, 2.91),     (-1.044, 3.004),    (-0.91, 3.095),
        (-0.782, 3.229),    (-0.714, 3.346),    (-0.698, 3.417),
        (-0.714, 3.513),    (-0.808, 3.655),    (-0.89, 3.72),
        (-0.976, 3.763),    (-1.064, 3.797),    (-1.166, 3.829),
        (-1.256, 3.845),    (-1.496, 3.848),    (-10.154, 3.886),
        (-10.294, 3.586),   (-10.904, 3.456),   (-9.106, 2.9),
        (-5.93, 2.983),     (-3.31, 0.876),     (-4.792, 0.613),
        (-5.044, 0.559),    (-5.198, 0.437),    (-5.286, 0.326),
        (-5.35, 0.198),     (-5.376, 0.097),    (-5.384, -0.000741),
        (-5.384, 0.0007),   (-5.376, -0.097),   (-5.35, -0.198),
        (-5.286, -0.326),   (-5.198, -0.437),   (-5.044, -0.559),
        (-4.792, -0.613),   (-3.31, -0.876),    (-5.93, -2.983),
        (-9.106, -2.9),     (-10.904, -3.456),  (-10.294, -3.586),
        (-10.154, -3.886),  (-1.496, -3.848),   (-1.256, -3.845),
        (-1.166, -3.829),   (-1.064, -3.797),   (-0.976, -3.763),
        (-0.89, -3.72),     (-0.808, -3.655),   (-0.714, -3.513),
        (-0.698, -3.417),   (-0.714, -3.346),   (-0.782, -3.229),
        (-0.91, -3.095),    (-1.044, -3.004),   (-1.336, -2.91),
        (-3.832, -2.92),    (-2.12, -0.973),    (0.022, -1.056),
        (0.03, -1.158),     (0.06, -1.287),     (0.11, -1.472),
        (0.168, -1.679),    (0.234, -1.862),    (0.302, -2.05),
        (0.348, -2.168),    (0.39, -2.268),     (0.48, -2.459),
        (0.696, -2.856),    (0.904, -3.165),    (0.984, -3.273),
        (1.08, -3.391),     (1.142, -3.468),    (1.24, -3.572),
        (1.37, -3.71),      (1.508, -3.841),    (1.706, -4.01),
        (1.912, -4.167),    (2.092, -4.287),    (2.248, -4.382),
        (2.516, -4.524),    (2.678, -4.601),    (2.836, -4.667),
        (3.112, -4.763),    (3.292, -4.814),    (3.508, -4.869),
        (3.814, -4.919),    (4.11, -4.95),      (4.286, -4.96),
        (4.498, -4.961),    (4.708, -4.955),    (4.988, -4.928),
        (5.144, -4.911),    (5.34, -4.879),     (5.55, -4.83),
        (5.872, -4.739),    (6.152, -4.637),    (6.354, -4.553),
        (6.544, -4.459),    (6.808, -4.318),    (7.024, -4.171),
        (7.238, -4.019),    (7.504, -3.798),    (7.68, -3.632),
        (7.872, -3.426),    (7.984, -3.298),    (8.118, -3.116),
        (8.262, -2.915),    (8.422, -2.654),    (8.542, -2.429),
        (8.646, -2.201),    (8.692, -2.088),    (8.74, -1.961),
        (8.824, -1.725),    (8.906, -1.423),    (8.942, -1.27),
        (8.984, -1.072),    (9.02, -0.829),     (9.052, -0.582),
        (9.064, -0.457),    (9.072, -0.29),     (9.08, 0.0003738)]

