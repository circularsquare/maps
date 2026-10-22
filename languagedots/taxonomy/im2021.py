"""Isle of Man, 2021 census Table 2.11, language mainly spoken at home (sources/im_census.py).

One answer per person (Q6: "What language do you use most in the home?"). Every printed language
has its own node; "Chinese" is the report's "all Chinese languages" (footnote 13), so the Sinitic
group; "Filipino" is the label printed, so the Filipino node, not the Philippine group; "Other"
(1,035) is the unnamed remainder, on `other`. Manx is not a row: it is inside "Other" if anyone
mainly speaks it at home (sources/im.md).
"""

NAMES = {
    "English": "indoeuropean.germanic.english",
    "Polish": "indoeuropean.slavic.west.polish",
    "Filipino": "austronesian.philippine.filipino",
    "Bulgarian": "indoeuropean.slavic.south.bulgarian",
    "Chinese": "sinotibetan.sinitic",
    "Afrikaans": "indoeuropean.germanic.continental.afrikaans",
    "Hungarian": "uralic.hungarian",
    "Portuguese": "indoeuropean.romance.portuguese",
    "Romanian": "indoeuropean.romance.romanian",
    "Spanish": "indoeuropean.romance.spanish",
    "Italian": "indoeuropean.romance.italian",
    "Other": "other",
}


def resolve(label):
    return NAMES[label]
