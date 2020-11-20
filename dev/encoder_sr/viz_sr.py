from plotly.subplots import make_subplots
import plotly.graph_objects as go

import numpy as np
import pandas as pd

def get_residual_df(df, time_index = 0):
    df = df[df['time'] == time_index]
    result = pd.merge(
        df[df['bool_predicted'] == 1][['x', 'y', 'z']], 
        df[df['bool_predicted'] == 0][['x', 'y', 'z']], 
        how='inner', on = ['x', 'y']
    )

    result['residual'] = np.abs(result['z_x'] - result['z_y'])
    
    return result


def plot_gaussian_ring(df, time_index = 0):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Actual', 'Predicted'))
    df = df[df['time'] == time_index]

    fig.add_trace(
        go.Scatter(
            x = df[df['bool_predicted'] == 0]['x'], 
            y = df[df['bool_predicted'] == 0]['y'], 
            mode = 'markers', 
            marker = dict(
                color = df[df['bool_predicted'] == 0]['z'],
                cmin = 0.5,
                cmax = 1.3
            )
        ),
        row = 1, col = 1
    )

    fig.add_trace(
        go.Scatter(
            x = df[df['bool_predicted'] == 1]['x'], 
            y = df[df['bool_predicted'] == 1]['y'], 
            mode = 'markers',
            marker = dict(
                color = df[df['bool_predicted'] == 1]['z'],
                cmin = 0.5,
                cmax = 1.3
            )
        ),
        row = 1, col = 2
    )

    fig.show()


def plot_residual_ring(df, time_index = 0):
    result = get_residual_df(df, time_index)

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Scatter(x=result['x'], y=result['y'], mode = 'markers', marker=dict(
            color=result['residual']
        )),
        row=1, col=1
    )

    fig.show()


def plot_gaussian_3d(df, time_index = 0):
    df_1 = df[df['time'] == time_index]

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])
    fig.add_trace(
        go.Scatter3d(
            x = df_1[df_1['bool_predicted'] == 0]['x'], 
            y = df_1[df_1['bool_predicted'] == 0]['y'], 
            z = df_1[df_1['bool_predicted'] == 0]['z'],
            mode='markers',
            marker=dict(
                size=2,
                color='blue',
                opacity = 0.6
            )
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter3d(
            x = df_1[df_1['bool_predicted'] == 1]['x'], 
            y = df_1[df_1['bool_predicted'] == 1]['y'], 
            z = df_1[df_1['bool_predicted'] == 1]['z'],
            mode='markers',
            marker=dict(
                size=2,
                color='blue',
                opacity = 0.6
            )
        ),
        row=1, col=2
    )

    fig.show()


def animate_gaussian_ring(df):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Actual', 'Predicted'))
    n_times = len(list(df['time'].unique()))
    
    df_init = df[df['time'] == 0]
    
    fig.add_trace(
        go.Scatter(
            x = df_init[df_init['bool_predicted'] == 0]['x'], 
            y = df_init[df_init['bool_predicted'] == 0]['y'], 
            mode = 'markers', 
            marker = dict(
                color = df_init[df_init['bool_predicted'] == 0]['z'],
                cmin = 0.5,
                cmax = 1.3
            )
        ),
        row = 1, col = 1
    )

    fig.add_trace(
        go.Scatter(
            x = df_init[df_init['bool_predicted'] == 1]['x'], 
            y = df_init[df_init['bool_predicted'] == 1]['y'], 
            mode = 'markers',
            marker = dict(
                color = df_init[df_init['bool_predicted'] == 1]['z'],
                cmin = 0.5,
                cmax = 1.3
            )
        ),
        row = 1, col = 2
    )

    frames = [go.Frame(
        data = [
            go.Scatter(
                x = df[(df['bool_predicted'] == 0) & (df['time'] == k)]['x'], 
                y = df[(df['bool_predicted'] == 0) & (df['time'] == k)]['y'], 
                mode = 'markers', 
                marker = dict(
                    color = df[(df['bool_predicted'] == 0) & (df['time'] == k)]['z'],
                    cmin = 0.5,
                    cmax = 1.3
                )
            ), 
            go.Scatter(
                x = df[(df['bool_predicted'] == 1) & (df['time'] == k)]['x'], 
                y = df[(df['bool_predicted'] == 1) & (df['time'] == k)]['y'], 
                mode = 'markers',
                marker = dict(
                    color = df[(df['bool_predicted'] == 1) & (df['time'] == k)]['z'],
                    cmin = 0.5,
                    cmax = 1.3
                )
            )
        ],
        traces=[0, 1]
    ) for k in range(n_times)]

    fig.frames = frames
    button = dict(
                 label='Play',
                 method='animate',
                 args=[None, dict(frame=dict(duration=50, redraw=False), 
                                  transition=dict(duration=0),
                                  fromcurrent=True,
                                  mode='immediate')]
    )
    
    fig.update_layout(updatemenus=[dict(type='buttons',
                                  showactive=True,
                                  y=0,
                                  x=1.05,
                                  xanchor='left',
                                  yanchor='bottom',
                                  buttons=[button] )
                                          ]
    )
    
    fig.show()

