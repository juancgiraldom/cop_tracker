#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import dash_bootstrap_components as dbc
from dash_extensions.enrich import DashProxy, MultiplexerTransform
import dash_leaflet as dl
from dash import html, dcc
import plotly.express as px
from dash.dependencies import Input, Output, State
import random
import pickle
import webbrowser

import pathlib

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.resolve()

# In[2]:


PRIVATE_JET_OCUP = 50 #Estimated based on 
PRIVATE_JET_EF = 3.044726409584053 # 4.9 kg per mile / 1.60934 km/mile(https://flybitlux.com/what-is-the-carbon-footprint-of-a-private-jet/#:~:text=A%20typical%20private%20jet%20emits,grams%20per%20passenger%20per%20kilometer.)


# In[3]:


emission_comparisson = [
    (1, ' times the CO2 an average European emits in a month'), 
    (1, ' flights from London to New York'), 
    (1961, ' vegetarian meals'), 
    (138, ' meat-based meals'),
    (1 ,' years of trash produced by a household in Canada'), 
    (4, ' months of energy for heating a home in Canada'),
    (72, ' trains from Amsterdam to Paris')
]


# In[4]:


def extract_colors(palette_name, n_colors, return_rgb=True):
    try:
        # Get the colormap
        cmap = plt.get_cmap(palette_name)
        
        # Generate n equally spaced values between 0 and 1
        values = np.linspace(0, 1, n_colors)
        
        # Get the colors from the colormap at the specified values
        rgb_colors = cmap(values)
        if return_rgb:
            return rgb_colors
        else:
            # Convert RGB to hex
            hex_colors = [to_hex(color) for color in rgb_colors]
            return hex_colors
        
    except ValueError as e:
        print(f"Error: {e}")
        return None

def create_line_chart(df, x, y,text=None):

    # Create the figure with Plotly Express
    fig=px.line(
        df, 
        x=x, 
        y=y, 
        labels={'value': 'Flight Emissions due to COP (ton CO2)', 'variable': 'Scenario'},
        color_discrete_map={'Historic':cop_colors['COP27'],'Projected': cop_colors['COP25']},
        markers=True,
        text=text
    )

    # Update the layout of the figure only
    fig.update_layout(
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),  # Set orientation to horizontal and adjust position
        margin=dict(l=10, r=10, t=30, b=10),  # Adjust margin/padding
        font=dict(family="Montserrat", color='white',size=8),  # Adjust font
        paper_bgcolor='#212222',
        
    )
    
    if text is not None:
        fig.update_traces(textposition='top center', textfont=dict(size=10,color=colorscale[0]))
    
    return fig


# In[11]:


with open(DATA_PATH.joinpath('assets/imagine_local_cops.pkl'), 'rb') as fp:
    imagine_local_cops = pickle.load(fp)
    print('imagine_local_cops')

with open(DATA_PATH.joinpath('assets/cops_proj.pkl'), 'rb') as fp:
    cops_proj = pickle.load(fp)
    print('cops_proj')

SELEC_COP = 'COP28'
VIS_PRIOR = 10


# In[12]:


cops_proj_gr = cops_proj[(cops_proj['Hierarchy']<=VIS_PRIOR)].groupby(['COP','Year']).agg({'FlightCO2Em':'sum'}).reset_index().rename(columns={'FlightCO2Em':'Projected'})
cops_proj_gr['Projected'] = cops_proj_gr['Projected'].astype(int)

cops_proj_gr['Historic'] = cops_proj_gr.apply(lambda row: row['Projected'] if row['Year'] <= 2023 else np.nan, axis=1)
cops_proj_gr['Projected'] = cops_proj_gr.apply(lambda row: row['Projected'] if row['Year'] >= 2023 else np.nan, axis=1)

centroid = cops_proj.total_bounds
centroid_x = (centroid[0] + centroid[2]) / 2
centroid_y = (centroid[1] + centroid[3]) / 2
centroid = [centroid_y,centroid_x]

hostdict = cops_proj.groupby('COP').agg({'HostCity':'first','HostCountry':'first', 'Year':'first'}).T.to_dict()
optimal_point, pois, gdf_edges = imagine_local_cops[hostdict[SELEC_COP]['HostCountry']]

centroid_local = optimal_point.total_bounds
centroid_x_local = (centroid_local[0] + centroid_local[2]) / 2
centroid_y_local = (centroid_local[1] + centroid_local[3]) / 2
centroid_local = [centroid_y_local,centroid_x_local]

TOT_KM = "{:.0f} MM km".format(cops_proj[(cops_proj['COP']==SELEC_COP)&(cops_proj['Hierarchy']<=VIS_PRIOR)]['Distance'].sum() / 1000000)
TOT_EMISSIONS = "{:.0f} TON OF CO2".format(cops_proj[(cops_proj['COP']==SELEC_COP)&(cops_proj['Hierarchy']<=VIS_PRIOR)]['FlightCO2Em'].sum())

RAND_IND = random.randint(0,len(emission_comparisson)-1)

EQ_EMISSIONS = "equivalent to {:.0f}".format(
    (cops_proj[(cops_proj['COP']==SELEC_COP)&(cops_proj['Hierarchy']<=VIS_PRIOR)]['FlightCO2Em'].sum())*emission_comparisson[RAND_IND][0]
) + emission_comparisson[RAND_IND][1]

ranking = cops_proj[
    (cops_proj['COP']==SELEC_COP)&(cops_proj['Hierarchy']<=VIS_PRIOR)
].groupby('Hierarchy').agg({'GuestCountry':'first'}).reset_index().rename(
    columns={'Hierarchy':'Pos','GuestCountry':f'{SELEC_COP} Guests'}
)


# In[13]:


app = DashProxy(
    transforms=[MultiplexerTransform()], 
    external_stylesheets=[dbc.themes.DARKLY],
    external_scripts=[
        {'src': 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.0/html2canvas.min.js'}
    ]
)

app.title= 'COP Tracker ✈️'

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "fontSize":"80%",
    "lineheight":"80%",
    "overflow": "scroll"
}

CONTENT_STYLE = {
    "margin-left": "20rem"
}

colorscale = ['#0b090a', '#161a1d', '#660708', '#a4161a', '#e5383b', '#b1a7a6', '#d3d3d3', '#f5f3f4']

cop_colors = extract_colors('Reds',len(hostdict.keys()),return_rgb=False)
cop_colors = {list(hostdict.keys())[i]:cop_colors[i] for i in range(len(list(hostdict.keys())))}
cop_colors['COP29'] = '#74D0E7'
cop_colors['COP30'] = '#219ebc'

poi_markers = {
    'Hotels':'https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Location_dot_red.svg/768px-Location_dot_red.svg.png',
    'Recycling':'https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Location_dot_blue.svg/768px-Location_dot_blue.svg.png',
    'Urban Gardens':'https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Location_dot_orange.svg/768px-Location_dot_orange.svg.png'
}

basemap = [
    dl.TileLayer(
        url='https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png',
        attribution='&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',

    #     url = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    #     attribution = 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
        maxZoom=16
    ),
]

#PREPARE GLOBAL MAP
temp_map = basemap.copy()

cops_proj[(cops_proj['Hierarchy']<=VIS_PRIOR)&(cops_proj['COP']==SELEC_COP)].explode().reset_index().apply(
        lambda row: temp_map.append(
            dl.Polyline(
                positions=[(coord[1], coord[0]) for coord in row['geometry'].coords],
                color='#D52221',
                weight=2
            )
        ),axis=1
)

#PREPARE LOCAL MAP
temp_map2 = basemap.copy()

gdf_edges[gdf_edges['width']!=0].apply(
    lambda row: temp_map2.append(
        dl.Polyline(
            positions=[(coord[1], coord[0]) for coord in row['geometry'].coords],
            color='#D52221',
            weight=0.2+(row['width']*10)
        )
    ),axis=1
)

optimal_point.geometry.apply(
    lambda geom: temp_map2.append(dl.Marker(
        position=[geom.y,geom.x],
        icon=dict(
            iconUrl='https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Dot-white.svg/2048px-Dot-white.svg.png',
            iconSize=[20, 20],
        )
    ))
)

pois.apply(
    lambda row: temp_map2.append(dl.Marker(
        position=[row['geometry'].y,row['geometry'].x],
        icon=dict(
            iconUrl=poi_markers[row['Type']],
            iconSize=[5, 5],
        )
    )),axis=1
)

viewport = dict(center=centroid_local)

#Generate Graph
fig = create_line_chart(cops_proj_gr, x='Year', y=['Historic','Projected'],text='COP')

app.layout = dbc.Container([
    dbc.Row(
        [
            dbc.Row(
                dbc.Label(' COP Tracker ✈️', style={'color':'white'}),
                style={'backgroundColor':colorscale[0],'fontSize':'100%','fontWeight':'bold','padding':'0px 0px 0px 10px'}
            ),
            dbc.Row([
                    dbc.Col([
                        dbc.Label('Choose a COP', style={'color':'white','fontSize':'70%','fontWeight':'bold','padding':'0px 0px 0px 10px'}),
                        dcc.Dropdown(
                            id='checklist-countries',
                            options=[
                                {
                                    'value':cop,
                                    'label':html.Span([f"{cop} ({hostdict[cop]['HostCity']}, {hostdict[cop]['HostCountry']})"],style={'color':'black'}),
                                } for cop in list(hostdict.keys())
                            ],
                            value=SELEC_COP,
                            multi=False,
                            placeholder='COP',
                            style={'fontSize':'90%'}
                        )
                    ], style={'backgroundColor':colorscale[1]},width=4), 
                    dbc.Col([
                        dbc.Label('Most Critical Guests', style={'color':'white','fontSize':'70%','fontWeight':'bold','padding':'0px 0px 0px 10px'}),
                        dcc.Slider(0, len(cops_proj['GuestCountry'].unique()),value=10,id='slider-prior',tooltip={"placement": "bottom", "always_visible": False}),
                    ], style={'backgroundColor':colorscale[1]},width=5),
                    dbc.Col([
                        dbc.Row(html.Br(id="redirect")),
                        dbc.Col(dbc.Button("Go to GitHub Repository",id='github-button'),width={'size': 8, 'offset': 2})
                    ], style={'backgroundColor':colorscale[1]})
                ],style={'padding':'0px 0px 0px 0px'}
            ),
        ],
        className='sticky-top fluid',style={'padding':'0px 0px 0px 10px','backgroundColor':colorscale[0]}
    ),
    dbc.Row([
        dbc.Col(dl.MapContainer(
            id='map-historic',
            children=temp_map,
            style={
                'width': '100%', 
                'height': '600px'
            },
            center=centroid,
            zoom=2
        ),width=4),
        dbc.Col([
            dbc.Row(html.Div(html.Div(id='ranking-table',
            children=dbc.Table.from_dataframe(
                ranking, striped=True, bordered=True, hover=True,style={'fontSize':'10px'},color='dark'
            ),style={'padding-left':'15px',"maxHeight": "375px", "overflow": "scroll"}),style={'height':'380'})),
            html.Br(),
            dbc.Row(dbc.Label(f'The {VIS_PRIOR} Most Critical Guests', id='top-emissions',style={'color':'white','fontSize':'80%','padding-left':'15px','text-align':'center'})),
            dbc.Row(dbc.Label(f'at {SELEC_COP}', id='cop-emissions',style={'color':'white','fontSize':'120%','padding-left':'15px','text-align':'center'})),
            dbc.Row(dbc.Label(f'travelled {TOT_KM}', id='val-km',style={'color':'white','fontSize':'100%','fontWeight':'bold','padding-left':'15px','text-align':'center'})),
            dbc.Row(dbc.Label(f'generating {TOT_EMISSIONS}', id='val-emissions',style={'color':'white','fontSize':'100%','fontWeight':'bold','padding-left':'15px','text-align':'center'})),
            dbc.Row(dbc.Label(EQ_EMISSIONS, id='eq-emissions',style={'color':'white','fontSize':'80%','padding-left':'15px','text-align':'center'})),
        ],width=2),
        dbc.Col(dcc.Graph(
            id='graph-projections',
            figure=fig,
            config={'displayModeBar': False},
            style={
                'width': '100%', 
                'height': '600px',
                'padding-left':'15px',
                'padding-right':'15px',
                'backgroundColor':'#212222'
            }
        ),width=3),
        dbc.Col([
            dbc.Row(dl.MapContainer(
                id='map-local',
                children=temp_map2,
                style={
                    'width': '100%', 
                    'height': '550px'
                },
                center=centroid_local,
                trackViewport=False,
                zoom=12
            )),
            dbc.Row([
                dbc.Col([
                    html.Img(src=poi_markers[poi], height='5px', width='5px', style={'objectFit': 'contain'}),
                    dbc.Label(poi, style={'color': 'white', 'fontSize': '45%'})
                ],style={'height':'15px'},width=2 if poi=='Hotels' else 3) for poi in poi_markers
            ]+[
                dbc.Col([
                    html.Img(src='https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Dot-white.svg/2048px-Dot-white.svg.png', height='7px', width='7px', style={'objectFit': 'contain'}),
                    dbc.Label('COP Venue', id='cop-venue-label', style={'color': 'white', 'fontSize': '45%'})
                ],style={'height':'15px'},width=4)
            ],style={'padding':'0px 0px 0px 0px'})  
        ],width=3,style={'backgroundColor':'#212222'}),
    ],className="g-0",style={'backgroundColor':'#212222'}),
],fluid=True,style={'fontFamily':'Montserrat','backgroundColor':'#212222'})

@app.callback(
    [
        Output('map-historic','children'),
        Output('map-local','children'),
        Output('map-local','viewport'),
        Output('cop-venue-label','children'),
        Output('ranking-table','children'),
        Output('top-emissions','children'),
        Output('cop-emissions','children'),
        Output('val-km','children'),
        Output('val-emissions','children'),
        Output('eq-emissions','children'),
        Output('graph-projections','figure')
    ],
    [
        Input('slider-prior','value'),
        Input('checklist-countries','value'),
    ]
)

def update_prioritization(vis_prior, selec_cop):
    global temp_map, temp_map2, SELEC_COP, VIS_PRIOR, optimal_point, pois, gdf_edges, centroid_local, cops_proj_gr

    VIS_PRIOR = vis_prior
    temp_map = basemap.copy()

    cops_proj[(cops_proj['Hierarchy']<=VIS_PRIOR)&(cops_proj['COP']==selec_cop)].explode().reset_index().apply(
            lambda row: temp_map.append(
                dl.Polyline(
                    positions=[(coord[1], coord[0]) for coord in row['geometry'].coords],
                    color='#D52221',
                    weight=2
                )
            ),axis=1
    )
    
    ranking = cops_proj[
        (cops_proj['COP']==selec_cop)&(cops_proj['Hierarchy']<=vis_prior)
    ].groupby('Hierarchy').agg({'GuestCountry':'first'}).reset_index().rename(
        columns={'Hierarchy':'Pos','GuestCountry':f'{selec_cop} Guests'}
    )
    
    ranking = dbc.Table.from_dataframe(
        ranking, striped=True, bordered=True, hover=True,style={'fontSize':'10px'},color='dark'
    )
    
    TOT_KM = "{:.0f} MM km".format(cops_proj[(cops_proj['COP']==selec_cop)&(cops_proj['Hierarchy']<=vis_prior)]['Distance'].sum() / 1000000)
    TOT_EMISSIONS = "{:.0f} ton CO2".format(cops_proj[(cops_proj['COP']==selec_cop)&(cops_proj['Hierarchy']<=vis_prior)]['FlightCO2Em'].sum())
    
    RAND_IND = random.randint(0,len(emission_comparisson)-1)

    EQ_EMISSIONS = "equivalent to {:.0f}".format(
        (cops_proj[(cops_proj['COP']==selec_cop)&(cops_proj['Hierarchy']<=vis_prior)]['FlightCO2Em'].sum())*emission_comparisson[RAND_IND][0]
    ) + emission_comparisson[RAND_IND][1]
    
    cops_proj_gr = cops_proj[(cops_proj['Hierarchy']<=vis_prior)].groupby(['COP','Year']).agg({'FlightCO2Em':'sum'}).reset_index().rename(columns={'FlightCO2Em':'Projected'})
    cops_proj_gr['Projected'] = cops_proj_gr['Projected'].astype(int)

    cops_proj_gr['Historic'] = cops_proj_gr.apply(lambda row: row['Projected'] if row['Year'] <= 2023 else np.nan, axis=1)
    cops_proj_gr['Projected'] = cops_proj_gr.apply(lambda row: row['Projected'] if row['Year'] >= 2023 else np.nan, axis=1)
    
    fig = create_line_chart(cops_proj_gr, x='Year', y=['Historic','Projected'],text='COP')

    if SELEC_COP != selec_cop:
        
        SELEC_COP = selec_cop
        
        optimal_point, pois, gdf_edges = imagine_local_cops[hostdict[SELEC_COP]['HostCountry']]

        centroid_local = optimal_point.total_bounds
        centroid_x_local = (centroid_local[0] + centroid_local[2]) / 2
        centroid_y_local = (centroid_local[1] + centroid_local[3]) / 2
        centroid_local = [centroid_y_local,centroid_x_local]
    
        temp_map2 = basemap.copy()

        gdf_edges[gdf_edges['width']!=0].apply(
            lambda row: temp_map2.append(
                dl.Polyline(
                    positions=[(coord[1], coord[0]) for coord in row['geometry'].coords],
                    color='#D52221',
                    weight=0.2+(row['width']*10)
                )
            ),axis=1
        )

        optimal_point.geometry.apply(
            lambda geom: temp_map2.append(dl.Marker(
                position=[geom.y,geom.x],
                icon=dict(
                    iconUrl='https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Dot-white.svg/2048px-Dot-white.svg.png',
                    iconSize=[20, 20],
                )
            ))
        )
        
    pois.apply(
        lambda row: temp_map2.append(dl.Marker(
            position=[row['geometry'].y,row['geometry'].x],
            icon=dict(
                iconUrl=poi_markers[row['Type']],
                iconSize=[5, 5],
            )
        )),axis=1
    )

    viewport = dict(center=centroid_local)
    
    map_text = 'COP Venue' if hostdict[selec_cop]['Year']<2024 else 'COP Venue (Hypothetical)'
    
    num_guests = f'The {vis_prior} Most Critical Guests' if vis_prior <195 else f'The {vis_prior} Guests'
    
    km_travelled = f'travelled {TOT_KM}' if hostdict[selec_cop]['Year']<2024 else f'will travel {TOT_KM}'
    
    return temp_map, temp_map2, viewport, map_text, ranking, num_guests, f'at {selec_cop}', km_travelled, f'generating {TOT_EMISSIONS}', EQ_EMISSIONS, fig


@app.callback(
    Output('redirect', 'children'),
    [Input('github-button', 'n_clicks')]
)
def redirect_to_github(n_clicks):
    if n_clicks is not None:
        webbrowser.open_new_tab("https://github.com/juancgiraldom/cop_tracker")
        return dcc.Location(pathname='/', id='dummy')
    else:
        return 

server = app.server
if __name__ == "__main__":
    app.run_server(debug=False, port=8080)



