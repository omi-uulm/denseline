import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from PIL import ImageDraw, Image

# constant values
VALUE = 'VALUE';
TIME = 'TIME';


def normalize_curve(data):
    """
    normalizes a drawn line
    :param data:
    :return:
    """
    xSum = data.sum(axis=0)
    # divide data with xSum ignoring the zeros
    return np.divide(data, xSum, where=xSum != 0)


def scale_values_to_image_size(array: np.array, extent, range: int):
    """
    scales an Array to an Range
    :param array: is an 1-dimensional numpy array
    :param range: is the range [1:range]
    :return:
    """
    min = extent[0]
    max = extent[1]
    scale = np.vectorize(lambda v: ((v - min) / (max - min)) * (range - 1))
    return scale(array)


def raster_curve(drawTuple, x_values, curve, normalize):
    """
    draws one curve and normalizes it

    :param drawTuple: is an tuple contains the Image and the ImageDraw Module (img, draw)
    :param curve: is an 1-dimensional array, which contains all points of the current curve
    :return:
    """

    prev_x = None
    prev_y = None
    size = drawTuple[0].size
    for i in range(len(curve)):

        record_x = x_values[i] - 1
        record_y = curve[i]

        if prev_x is not None and prev_y is not None:
            drawTuple[1].line([(prev_x, prev_y), (record_x, record_y)])
        prev_x = record_x
        prev_y = record_y

    if normalize:
        series_data = normalize_curve(np.array(drawTuple[0]))
    else:
        series_data = np.array(drawTuple[0])

    # reset Image
    drawTuple[1].rectangle([(0, 0), size], fill=0)
    return series_data


def create_image_draw_tuple(size: (int, int)) -> (Image, ImageDraw):
    """
    creates a tuple, which contains the drawModule and the image
    :param height: is the height of the image
    :param width: is the width of the image
    :return: an dictionary (img, draw). img is the Image, draw is the ImageDraw-Module
    """
    img = Image.new('I', size, color=0)
    draw = ImageDraw.Draw(img)
    imageTuple = (img, draw)
    return imageTuple


def calc_extents(value, time):
    # compute x and y extents
    y_extent = [value.min(), value.max()]
    x_extent = [time.min(), time.max()]

    return (x_extent, y_extent);


def compute_denseline(image_draw_tuple: (Image, ImageDraw), x_values, y_values, normalize) -> np.ndarray:
    """
    computes densline Matrix
    :param image_draw_tuple: is a tuple with type (Image, ImageDraw)
    :param x_values: is a digitized Timescale
    :param y_values: are the scaled y_values
    :return: is a matrix with values in range [0,1]
    """
    image_size = image_draw_tuple[0].size
    data = np.zeros(image_size, dtype=np.float32)

    for column in y_values:
        series_data = raster_curve(image_draw_tuple, x_values, column, normalize)
        data += series_data

    data = np.divide(data, len(y_values))
    return data


def digitize_x_values(time_extent,is_time, width, values) -> np.ndarray:
    """
    digitizes x_values, such as the timestamps, for the x_values
    :param time_extent: is a tuple, which contains the min and max values
    :param width: is the width of the image
    :param values: are the sorted values
    :return:

    """
    if not is_time:
        x_bins = np.linspace(time_extent[0], time_extent[1], num=width)
        time_scale = np.digitize(values, x_bins)
    else:
        x_bins = pd.date_range(time_extent[0], time_extent[1], periods=width)
        time_scale = np.digitize(values, x_bins.values)
    return time_scale


def plot_heat_map(data: np.ndarray, extends,isTime_serie,title, x_lable, y_lable) -> None:
    """
    plots the data on an image using the colormap viridis_r to show the density
    :param data: is the calculatet Matrix with the density range
    :param x_extend has to be an datetime start
    :param y_extend
    :return:
    """
    x_extend, y_extend = extends

    # plot the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # selecting the colormap
    mycmap = plt.get_cmap('viridis_r')
    mycmap.set_under('w')

    min_non_null = data[data > 0].min()

    im = ax.imshow(data, interpolation='nearest', cmap=mycmap, vmin=min_non_null,
                   aspect='auto', extent=[x_extend[0], x_extend[1], y_extend[0], y_extend[1]], origin='lower')
    cbar = fig.colorbar(im, fraction=0.03, pad=0.05)
    if (isTime_serie):
        ax.xaxis_date()
    ax.set_xlabel(x_lable)
    ax.set_ylabel(y_lable)
    ax.set_title(title)

    print("show pic")
    plt.tight_layout()

    fig.savefig('out/'+title+'.png', dpi=fig.dpi)

    return


def denseline(df: pd.DataFrame, height: int, width: int, normalize):
    """

    :param df: a pandas Dataframe.
    :param height:
    :param width:
    :return:
    """
    time = df.index.values
    value = df.values
    value = value.T
    # creating Image Matrix
    image_draw_tuple = create_image_draw_tuple(height, width)

    x_extent, y_extent = calc_extents(value, time)

    y_values = scale_values_to_image_size(value, y_extent, height)
    x_values = digitize_x_values(x_extent, width, time)

    # compute heatmap
    data = compute_denseline(image_draw_tuple, x_values, y_values, normalize)
    plot_heat_map(data, (x_extent, y_extent))
    return


# main methode to run denseline 
def main(df, size=(1000, 1000), title="Line Heatmap", x_lable="time", y_lable="value", normalize=True):
    """

    :param df: is a pandas Dataframe, which contains ONLY numbers as values. All columns will be used as a single series in the plot.
        The information, which timeseries is which cannot be displayed in the plot
        the df should be look like this:

    :param size: represents the resolution for the raster matrix
    :param title: is the title for the graph
    :param x_lable: the x_lable
    :param y_lable: the y_lable
    :return
    """
    # initialize values

    time = df.index
    isTime_series = isinstance(time, pd.DatetimeIndex)
    time = time.values
    # all values stored as np_array so it's easier for normalize the values
    value = df.values
    value = value.T
    width, height = size
    # creating Image Matrix
    image_draw_tuple = create_image_draw_tuple(size)
    x_extent, y_extent = calc_extents(value, time)
    # normalize the values
    y_values = scale_values_to_image_size(value, y_extent, height)
    x_values = digitize_x_values(x_extent,isTime_series, width, time)

    # compute heatmap
    dense_matrix = compute_denseline(image_draw_tuple, x_values, y_values, normalize)

    plot_heat_map(dense_matrix, (x_extent, y_extent),isTime_series,title, x_lable, y_lable)
    return

