import pandas as pd
import folium
import os
from folium import LayerControl
import numpy as np

def gen_map (final_df, date, time):
  # Partie 4 Construire la carte et mettre les points du dataframe dessus
  final_df['time'] = pd.to_datetime(final_df['time'], errors='coerce')

  # Combiner les informations de jour et avant/après 12h dans une seule colonne
  final_df['Day'] = final_df['time'].apply(lambda x: f"{x.date()} {'Matin' if x.hour < 12 else 'Apres mid'}")

  days = final_df['Day'].unique()

  # Générer une palette de 20 couleurs allant du vert au noire
  colors = ['#00FF00', '#32CD32', '#66FF00', '#99FF00', '#CCFF00', '#FFFF00', '#FFCC00', '#FF9900', '#FF6600', '#FF3300',
            '#FF0000', '#FF00FF', '#FF69B4', '#FF1493', '#C71585', '#8B008B', '#800080', '#4B0082', '#0000FF', '#000000']

  # Définir les bins pour 20 tranche de valeurs de part égale
  bins = np.arange(1, 21, 1)

  # Créer des groupes de calques pour chaque jour et chaque paramètre de polluant
  pollutants = {
      'PM1 (ug/m3)': 20,
      'PM2.5 (ug/m3)': 40,
      'PM4 (ug/m3)': 50,
      'PM7 (ug/m3)': 60,
      'PM10 (ug/m3)': 80,
      'TSP (ug/m3)': 100,
      'BC880 (ug/m3)': 20,
      'BC375 (ug/m3)': 20,
      'Blue BCc (ug/m3)': 20,
      'Red BCc (ug/m3)': 20,
      'Green BCc (ug/m3)': 20,
      '0.3 (/L)': 200000,
      '0.5 (/L)': 50000,
      '1.0 (/L)': 5000,
      '2.5 (/L)': 900,
      '4.0 (/L)': 250,
      '5.0 (/L)': 200,
      '7.0 (/L)': 100,
      '10 (/L)': 90, 
      'Fossil Fuel': 20, 
      'Biomass': 20, 
      'Number' : 100000
      # 'noise': 100,
      # 'voc' : ???
      # 'Increment PM2 5':30,
      # 'Increment PM10':80
  }

  incre = {
      'BC880 (ug/m3)': 20,
      'BC375 (ug/m3)': 20,
      # 'Increment PM2 5':30,
      # 'Increment PM10':80
  }

  def sanitize_identifier(identifier):
      return identifier.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "").replace("/", "_")

  # Créer une carte centrée sur Lyon
  map = folium.Map(location=[45.75, 4.85], zoom_start=13)
  incremap = folium.Map(location=[45.75, 4.85], zoom_start=13)

  # Ajouter des couches de groupe pour chaque combinaison de jour et polluant
  legend_html = ""
  legend_js = ""

  # creation de la carte des polluants
  for day in days:
      day_df = final_df[final_df['Day'] == day]
      for pollutant, max_value in pollutants.items():
          feature_group = folium.FeatureGroup(name=f'{day} - {pollutant}', overlay=True, control=True, show=False)
          bins = np.linspace(0, max_value, 21)  # Créer des bins spécifiques au polluant
          for _, row in day_df.iterrows():
              value = row[pollutant]
              # Trouver l'index de la couleur correspondant à la valeur du polluant
              if value <= max_value:
                  bin_index = np.digitize(value, bins) - 1
                  bin_index = min(max(bin_index, 0), len(colors) - 1)  # Assurer que l'index est dans la plage valide
                  color = colors[bin_index]
              else:
                  color = colors[-1]  # Couleur noire pour les valeurs supÃ©rieures au maximum
              folium.CircleMarker(
                  location=(row['Latitude'], row['Longitude']),
                  radius=3,
                  popup=f"{pollutant}: {row[pollutant]} ug/m3\nDate: {row['time']}\nHeart Rate: {row['Heart Rate']}\nTemperature {row['T (C)']}",
                  fill=True,
                  color=color,
                  fill_opacity=0.6
              ).add_to(feature_group)
          feature_group.add_to(map)

          # Ajouter une légende pour chaque groupe
          colormap = folium.LinearColormap(colors=colors, vmin=0, vmax=max_value, caption=f'{pollutant}')
          colormap_html = colormap._repr_html_()

          safe_pollutant = sanitize_identifier(pollutant)
          safe_day = sanitize_identifier(str(day))

          legend_html += f'<div id="legend_{safe_day}_{safe_pollutant}" style="display: none;">{colormap_html}</div>'
          legend_js += f"""
          $('input:checkbox + span:contains("{day} - {pollutant}")').parent().find('input').change(function() {{
              if ($(this).is(':checked')) {{
                  $('#legend_{safe_day}_{safe_pollutant}').show();
              }} else {{
                  $('#legend_{safe_day}_{safe_pollutant}').hide();
              }}
          }});"""

  # Ajouter des controles de calque
  LayerControl(collapsed=False).add_to(map)

  # Ajouter des controles de calque
  LayerControl(collapsed=False).add_to(incremap)

  # Ajouter une légende dynamique avec JavaScript
  legend_html = f"""
  <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px; height: auto; 
                z-index:9999; font-size:14px;">
                {legend_html}
  </div>
  <script src="https://code.jquery.com/jquery-1.12.4.min.js"></script>
  <script>
  $(document).ready(function() {{
      {legend_js}
  }});
  </script>
  """

  map.get_root().html.add_child(folium.Element(legend_html))

  incremap.get_root().html.add_child(folium.Element(legend_html))

  # Enregistrer la carte dans un fichier HTML
  html_output_file = f'Cartes/Carte_{date}_{time}.html'
  map.save(html_output_file)
  # incremap.save('Carte Incrementation.html')

  # Vérifier si le fichier a été enregistré correctement

  if os.path.exists(html_output_file):
      print(f"Carte enregistrée dans {html_output_file}")
  else:
      print(f"Erreur lors de l'enregistrement de la carte {html_output_file}")