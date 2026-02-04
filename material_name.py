# import os
# import re
# import json
# from pymatgen.io.cif import CifParser

# plots_dir = 'plots'
# plot_files = [f for f in os.listdir(plots_dir) if f.endswith('_kkr_plot.png')]

# material_ids = []
# for filename in plot_files:
#     match = re.match(r'(mp-\d+)_kkr_plot\.png', filename)
#     if match:
#         material_ids.append(match.group(1))

# print(f"Found {len(material_ids)} materials in plots folder:")
# print(f"Material IDs: {material_ids}\n")

# print("Loading ENdiele20.json...")
# with open('ENdiele20.json', 'r') as f:
#     data = json.load(f)

# mpid_to_entry = {entry.get('mpid'): entry for entry in data if 'mpid' in entry}

# print("Material names (using jid + cif):")
# for material_id in material_ids:
#     entry = mpid_to_entry.get(material_id)
#     if not entry:
#         print(f"  {material_id}: Not found in JSON")
#         continue
    
#     jid = entry.get('jid', 'N/A')
#     cif = entry.get('cif', '')
    
#     if not cif:
#         print(f"  {material_id} (jid: {jid}): No CIF data")
#         continue
    
#     try:
#         parser = CifParser.from_str(cif)
#         structure = parser.get_structures()[0]
#         formula = structure.formula
#         print(f"  {material_id} (jid: {jid}): {formula}")
#     except Exception as e:
#         print(f"  {material_id} (jid: {jid}): Error parsing CIF - {str(e)}")


import os
import re
import json
from pymatgen.io.cif import CifParser

try:
    parser = CifParser.from_str(cif)
    structure = parser.get_structures()[0]
    formula = structure.formula
    print(f"  {material_id} (jid: {jid}): {formula}")
except Exception as e:
    print(f"  {material_id} (jid: {jid}): Error parsing CIF - {str(e)}")
