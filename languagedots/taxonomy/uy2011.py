"""Uruguay, Censos 2011: country of birth -> origin ISO code; each origin's languages come from
sources/origin_mix.py (mix(iso, "uy")), the Uruguayan-born ("UY") are Spanish. No language
question; every row `derived` (sources/uy.md).

COUNTRY maps PAISNAC's labels as INE prints them. Dissolved states take origin_mix's pseudo
codes (URSS SU, Yugoeslavia YU, Checoeslovaquia QT); Germany's two old states, the Canaries,
the UK's nations and the Channel Islands their present country. REMAINDER labels name no
birthplace: "No relevado" (115,797, people whose birthplace was not collected, not
foreign-born) and "No declarado o ignorado". countries/uy.py spreads them over the unit's known
birthplaces, Uruguay included, in proportion.
"""

COUNTRY = {
    'Argelia': 'DZ', 'Andorra': 'AD', 'Angola': 'AO', 'Argentina': 'AR', 'Australia': 'AU',
    'Austria': 'AT', 'Bahamas': 'BS', 'Bahréin': 'BH', 'Bangladesh': 'BD', 'Armenia': 'AM',
    'Barbados': 'BB', 'Bélgica': 'BE', 'Bermudas': 'BM', 'Bolivia': 'BO',
    'Bosnia Herzegovina': 'BA', 'Botswana': 'BW', 'Brasil': 'BR', 'Belice': 'BZ',
    'Islas Vírgenes Británicas': 'VG', 'Brunei Darussalam': 'BN', 'Bulgaria': 'BG',
    'Burundi': 'BI', 'Bielorrusia': 'BY', 'Camerún': 'CM', 'Canadá': 'CA', 'Cabo Verde': 'CV',
    'Islas Caymán': 'KY', 'Sri Lanka': 'LK', 'Chile': 'CL', 'China': 'CN', 'Colombia': 'CO',
    'Comoras': 'KM', 'Congo': 'CG', 'Congo, República Democrática del': 'CD',
    'Costa Rica': 'CR', 'Croacia': 'HR', 'Cuba': 'CU', 'Chipre': 'CY', 'República Checa': 'CZ',
    'Dinamarca': 'DK', 'Dominica': 'DM', 'República Dominicana': 'DO', 'Ecuador': 'EC',
    'El Salvador': 'SV', 'Etiopía': 'ET', 'Eritrea': 'ER', 'Estonia': 'EE',
    'Islas Feroe': 'FO', 'Fiji': 'FJ', 'Finlandia': 'FI', 'Francia': 'FR',
    'Polinesia Francesa': 'PF', 'Gabón': 'GA', 'Gambia': 'GM',
    'Palestinos, Territorios Ocupados': 'PS', 'Ghana': 'GH', 'Gibraltar': 'GI',
    'Grecia': 'GR', 'Granada': 'GD', 'Guadalupe': 'GP', 'Guam': 'GU', 'Guatemala': 'GT',
    'Guyana': 'GY', 'Haití': 'HT', 'Honduras': 'HN', 'Hong Kong': 'HK', 'Hungría': 'HU',
    'Islandia': 'IS', 'India': 'IN', 'Indonesia': 'ID', 'Irán': 'IR', 'Irak': 'IQ',
    'Irlanda': 'IE', 'Israel': 'IL', 'Italia': 'IT', 'Costa de Marfil': 'CI', 'Jamaica': 'JM',
    'Japón': 'JP', 'Kazakhstan': 'KZ', 'Jordania': 'JO', 'Kenia': 'KE', 'Corea del Sur': 'KR',
    'Kuwait': 'KW', 'Kyrgyzstan': 'KG', 'Líbano': 'LB', 'Letonia': 'LV', 'Libia': 'LY',
    'Lituania': 'LT', 'Luxemburgo': 'LU', 'Malawi': 'MW', 'Malasia': 'MY', 'México': 'MX',
    'Mónaco': 'MC', 'Moldavia': 'MD', 'Montserrat': 'MS', 'Marruecos': 'MA',
    'Mozambique': 'MZ', 'Omán': 'OM', 'Namibia': 'NA', 'Nepal': 'NP', 'Holanda': 'NL',
    'Antillas Holandesas': 'CW', 'Aruba': 'AW', 'Nueva Caledonia': 'NC',
    'Nueva Zelanda': 'NZ', 'Nicaragua': 'NI', 'Nigeria': 'NG', 'Noruega': 'NO',
    'Pakistán': 'PK', 'Panamá': 'PA', 'Paraguay': 'PY', 'Perú': 'PE', 'Filipinas': 'PH',
    'Polonia': 'PL', 'Portugal': 'PT', 'Puerto Rico': 'PR', 'Rumania': 'RO', 'Rusia': 'RU',
    'Santa Helena': 'SH', 'Anguilla': 'AI', 'San Vicente y las Granadinas': 'VC',
    'Arabia Saudita': 'SA', 'Senegal': 'SN', 'Serbia': 'RS', 'Singapur': 'SG',
    'Eslovaquia': 'SK', 'Eslovenia': 'SI', 'Somalia': 'SO', 'Sudáfrica': 'ZA',
    'Zimbabwe': 'ZW', 'Sahara Oeste': 'EH', 'Surinam': 'SR', 'Suecia': 'SE', 'Suiza': 'CH',
    'Siria': 'SY', 'Thailanda': 'TH', 'Trinidad y Tobago': 'TT',
    'Emiratos Árabes Unidos': 'AE', 'Túnez': 'TN', 'Turquía': 'TR', 'Uganda': 'UG',
    'Ucrania': 'UA', 'Macedonia': 'MK', 'Egipto': 'EG', 'Islas Channel': 'GB',
    'Tanzania': 'TZ', 'Estados Unidos de América': 'US', 'Uzbekistán': 'UZ',
    'Venezuela': 'VE', 'Zambia': 'ZM', 'Alemania': 'DE', 'Alemania Federal': 'DE',
    'Alemania, Rpca. Democrática': 'DE', 'España': 'ES', 'Islas Canarias': 'ES',
    'Reino Unido (Gran Bretaña e Irlanda del Norte)': 'GB', 'Inglaterra': 'GB',
    'Escocia': 'GB', 'Gales': 'GB', 'Checoeslovaquia': 'CZ', 'China Continental': 'CN',
    'China Taiwan': 'TW', 'Islas Pacifico de EEUU': 'US', 'Palestina (zona neutral)': 'PS',
    'URSS': 'SU', 'Yugoeslavia (Serbia y Montenegro)': 'YU',
}
COUNTRY['Checoeslovaquia'] = 'QT'
REMAINDER = {"No relevado", "No declarado o ignorado"}

SPANISH = "indoeuropean.romance.spanish"
OTHER = "other"


def mix(origin):
    """{node: share} for people born in `origin` (ISO, "UY" or "rest") living in Uruguay."""
    if origin == "UY":
        return {SPANISH: 1.0}
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sources"))
    import origin_mix
    return origin_mix.mix(origin, "uy")
