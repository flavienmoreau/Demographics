# Title:  Isocode mapping object and function
# Created By: Calvin He
# Date Created: 2023-06-21
# Description: 
# Output: 
# Notes: 


path_intermediate_data = "@Import/Data/intermediate_data/"

import pandas as pd

# Import mapping between country name and 3 letters isocode
isocode_list_full = ['AUS', 'CAN', 'DEU', 'GBR', 'ITA', 'SWE', 'AUT', 'BEL', 'ESP', 'EST', 'FIN', 'FRA', 'GRC',
                        'HUN', 'IRL', 'LUX', 'NLD', 'POL', 'SVK', 'SVN', 'CHN', 'IND', 'JPN', 'DNK', 'USA'] #, 'NGA']

isocode_2digit = ["AU", "CA", "DE", "GB", "IT", "SE", "AT", "BE", "ES", "EE", "FI", "FR", "GR", 
                  "HU", "IE","LU", "NL", "PL", "SK", "SI", "CN", "IN", "JP", "DK", "US"] #, "NG" ]

country = ["Australia", "Canada", "Germany", "United Kingdom", "Italy", "Sweden", "Austria", "Belgium", "Spain", "Estonia",
"Finland", "France", "Greece", "Hungary", "Ireland", "Luxembourg", "Netherlands", "Poland", "Slovakia", "Slovenia", "China", "India", 
"Japan", "Denmark", "United States of America"] #,"Nigeria"]

isocodes = pd.DataFrame({"isocode": isocode_list_full, "isocode_2digit": isocode_2digit, "country": country})

isocode_list_LICs = pd.read_excel(path_intermediate_data + "lic_list.xlsx", header = 0).drop('ifscode', axis=1).rename(columns={'iso3code': 'isocode'})
isocode_list_LICs = isocode_list_LICs['isocode'].tolist()



def UN_isocodes(LIC=False):
    
    if not LIC:
        
        # Import mapping between country name and 3 letters isocode
        isocode_list_full = ['AUS', 'CAN', 'DEU', 'GBR', 'ITA', 'SWE', 'AUT', 'BEL', 'ESP', 'EST', 'FIN', 'FRA', 'GRC',
                             'HUN', 'IRL', 'LUX', 'NLD', 'POL', 'SVK', 'SVN', 'CHN', 'IND', 'JPN', 'DNK', 'USA'] #,'NGA']
    
        country = ["Australia", "Canada", "Germany", "United Kingdom", "Italy", "Sweden", "Austria", "Belgium", "Spain", "Estonia",
        "Finland", "France", "Greece", "Hungary", "Ireland", "Luxembourg", "Netherlands", "Poland", "Slovakia", "Slovenia", "China", "India", 
        "Japan", "Denmark", "United States of America"] #,"Nigeria"]
        
        isocodes = pd.DataFrame({"isocode": isocode_list_full, "country": country})
        
    else:
        
        path_intermediate_data = "@Import/Data/intermediate_data/"
        isocodes = pd.read_excel(path_intermediate_data + "lic_list.xlsx", header = 0)
        isocodes= isocodes.drop('ifscode', axis=1)
        isocodes= isocodes.rename(columns={'iso3code': 'isocode'})

    return isocodes