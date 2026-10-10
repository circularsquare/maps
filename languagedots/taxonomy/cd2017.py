"""DR Congo, MICS-Palu 2017-18 (sources/cd_mics.py): the household head's mother tongue (HC1B) ->
node. MICS names the four national languages and French; "another language" is split among the
local languages by the Enquete 1-2-3 ethnic model, whose labels taxonomy/cd2012.py maps (resolve()
falls through to it), and so is Kikongo where the head is ethnic Kongo.

  * "Tshiluba" is Luba-Kasai, the same node the ethnic model gives Luba and Lulua heads.
  * "Kikongo (Kikongo ya leta)": MICS's Kikongo where the ethnic model has no Kongo heads to
    carry it, nearly all in Kwilu and Kwango (11-15% of heads there, almost no ethnic Kongo): the
    vehicular Kikongo of the region, Kituba (Glottolog kitu1246 covers both banks), cg.txt's node.
    In Kinshasa and Kongo Central MICS's Kikongo is the Kongo varieties, by the ethnic model.
  * "Swahili" is Congo Swahili, zm.txt's node; "Lingala" ao.txt's.
"""
import cd2012

NAMES = {
    "French": "indoeuropean.romance.french",
    "Lingala": "nigercongo.bantu.lingala",
    "Swahili": "nigercongo.bantu.swahili_congo",
    "Tshiluba": cd2012.LUBA_KASAI,
    "Kikongo (Kikongo ya leta)": "nigercongo.bantu.kituba",
}


def resolve(label, province=None):
    return NAMES.get(label) or cd2012.resolve(label, province)
