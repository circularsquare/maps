from pathlib import Path

root = Path('C:/Users/anita/projects/citycircles')
aliases = """  // Bilingual city names and common short forms.
  'sansebastian':              'donostiasansebastian',
  'donostia':                  'donostiasansebastian',
  'donosti':                   'donostiasansebastian',
  'sudbury':                   'greatersudbury',
  'vitoria':                   'gasteizvitoria',
  'gasteiz':                   'gasteizvitoria',
  'vitoriagasteiz':            'gasteizvitoria',
  'arrasate':                  'arrasatemondragon',
  'mondragon':                 'arrasatemondragon',
  'laudio':                    'laudiollodio',
  'llodio':                    'laudiollodio',
  'biel':                      'bielbienne',
  'bienne':                    'bielbienne',
"""
registration = """  // Preserve same-name cities when adding an alias (e.g. Sudbury in CA/GB/US).
  for(const [alias,canonical] of Object.entries(CITY_ALIASES)){
    const entries=cityNameMap.get(canonical);
    if(!entries) continue;
    const existing=cityNameMap.get(alias)||[];
    cityNameMap.set(alias,[...existing,...entries.filter(e=>!existing.some(x=>x.i===e.i))]);
  }
"""
for name in ('citycircles.html', 'citycircles_globe.html'):
    path = root / name
    original = path.read_bytes()
    text = original.decode('utf-8').replace('\r\n', '\n')
    if name == 'citycircles.html':
        marker = 'const CITY_ALIASES={\n'
        assert text.count(marker) == 1
        text = text.replace(marker, marker + aliases)
        start = text.index('  // Register aliases')
        end = text.index('\n}', start)
        text = text[:start] + registration.rstrip('\n') + text[end:]
    else:
        marker = 'function loadCities(){'
        assert text.count(marker) == 1
        text = text.replace(marker, 'const CITY_ALIASES={\n' + aliases + '};\n\n' + marker)
        start = text.index(marker)
        end = text.index('\n}', start)
        text = text[:end] + '\n' + registration.rstrip('\n') + text[end:]
    if b'\r\n' in original:
        text = text.replace('\n', '\r\n')
    path.write_bytes(text.encode('utf-8'))

path = root / 'city_aliases.csv'
original = path.read_bytes()
addition = '''
# --- BILINGUAL NAMES AND COMMON SHORT FORMS ---

Donostia / San Sebasti\u00e1n,Spain,San Sebastian|Donostia|Donosti
Greater Sudbury,Canada,Sudbury
Gasteiz / Vitoria,Spain,Vitoria|Gasteiz|Vitoria-Gasteiz
Arrasate / Mondrag\u00f3n,Spain,Arrasate|Mondragon
Laudio / Llodio,Spain,Laudio|Llodio
Biel/Bienne,Switzerland,Biel|Bienne
'''
if b'\r\n' in original:
    addition = addition.replace('\n', '\r\n')
path.write_bytes(original + addition.encode('utf-8'))
