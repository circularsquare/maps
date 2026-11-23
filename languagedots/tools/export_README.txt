languagedots: dot data
======================

The data behind the map at https://anita.garden/languagedots/, first language (mother tongue or
language spoken at home) for 227 countries and territories, 7.39 billion people. Built 21
October 2026.


Files
-----

dots.csv        one row per mark on the map
languages.csv   the language tree: every id used in dots.csv, with its name and parent
countries.csv   for each country, the census or survey used and how fine it is


dots.csv
--------

lon, lat    WGS84, 4 decimal places
country     two-letter code (ISO 3166-1, except uk for the United Kingdom, xk for Kosovo and
            xs for the Israeli settlements in the West Bank, drawn apart from ps)
language    an id from languages.csv
people      1000 for an ordinary dot. A smaller number is a leftover: the people of that
            language in that census unit after the whole dots were taken out, or a language
            too small anywhere in the country to reach one dot. Summing people by country
            and language gives the census totals the map uses.

Where the language is a family or group id (it has children in languages.csv), the source
only says "other languages of this group" and does not name one. The id "other" is a
language the source does not identify at all.


Read this before mapping it
---------------------------

The dots are not where anyone lives. Each source gives counts per area (a district, a
province, sometimes only the whole country), and inside each area the dots are scattered at
random, weighted by population density. Any pattern finer than the source's own areas is
made up. countries.csv says how fine each country is in its "grain" column: about 6,000
sub-districts for India, the country as one unit for Andorra.

Sources differ in year, in the question asked (mother tongue, language at home, main
language) and in who was counted. Some countries come from surveys, not censuses, and carry
sampling error. The "how" and "note" columns say which.


Licence
-------

CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Please credit
"languagedots, anita.garden" and keep the same licence on data derived from it.

Some of the inputs have their own credit lines:
- Country sources are named in countries.csv.
- District language shares for Benin, Guinea, Mali, Senegal and Sierra Leone are from CLEAR
  Global (CC BY-SA 4.0), via the Humanitarian Data Exchange.
- Language names and families follow Glottolog (CC BY 4.0) where the census does not.
- Dots are placed with population density layers including Kontur Population (CC BY 4.0).
