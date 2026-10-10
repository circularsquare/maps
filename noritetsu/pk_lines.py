"""Pakistan's passenger line list for pk_register.py (pk_sources.md has the sources and what
runs). Each line is a run of track between two places, wholly running or wholly not, every
piece of track on exactly one line.

POINTS:
    "Name"              the OSM rail station of that name (name, name:en, alt names; case,
                        "Railway Station", "Junction"/"Jn" folded away) nearest the line's
                        previous point (the next one's, for a line's first point)
    "Name@lon,lat"      that station near the coordinate (pk_register.NEAR_M); none of the name
                        there: the nearest OSM rail station within BLIND_M, else left out
    "~Name@lon,lat"     a junction at the coordinate, no stop
    "#ID"               a border point (BORDERS)
  each optionally followed by "|km": Pakistan Railways' km post, from en.wikipedia's route
  diagram (RDT) of the line. A section between two points with posts takes their difference as
  its length (the chainage check_model reads); else its traced length. A point no OSM station
  matches is left out, and its neighbours' posts still measure the section across it.

`chain: True` marks a line whose every listed point carries a post: its km_official is shipped.
`suspended: True`: track kept, greyed, out of completion.
"""

PR = "Pakistan Railways"

# id -> (lon, lat, [cc, cc]). Koh-i-Taftan - Mirjaveh: where OSM's track (way 1457452999,
# usage=main) crosses OSM's Pakistan - Iran boundary (way 440799384), Overpass 2026-10-08,
# about 550 m west of Taftan station. Proposed for borders.EXTRA as "eXIRPK1"
# (handoff_notes/pk_build.md).
BORDERS = {
    "XIRPK1": (61.551797, 28.974554, ["ir", "pk"]),
}


def L(lid, name, pts, suspended=False, chain=False, name_en="", note="", more=()):
    return {"id": lid, "name": name, "name_en": name_en, "pts": list(pts), "more": list(more),
            "im": PR, "suspended": suspended, "chain": chain, "note": note}


LINES = [
    # ------------------------------------------------------------------ ML-1
    # Karachi - Peshawar, PR km posts from Kiamari (0). Main km run Lodhran - Khanewal by the
    # chord (Shahidanwala ... Mehar Shah, 938 at Khanewal), which the Allama Iqbal Express
    # runs; the line by Multan has its own posts (0 at Lodhran, 136 at Khanewal) and is its own
    # line below. Halts the RDT gives no post are left out (osm_stops puts OSM's stations on
    # the traced sections anyway); Serai Alamgir's "1,365" (between 1,381 and 1,389) is a
    # typo and left out, and Gujranwala City (1,290), which Wikidata places on Gujranwala.
    L("ml1", "Karachi–Peshawar Line", [
        "Karachi City|5", "Karachi Cantonment|9", "Drigh Road|19", "Drigh Colony|21",
        "Malir Colony|24", "Malir|26", "Landhi Junction|29", "Jummah Goth|35", "Bin Qasim|43",
        "Badal Nala|45", "Pipri|48", "Gaddar|51", "Dabheji|61", "Ran Pethani|79",
        "Jungshahi|91", "Braudabad|108", "Jhimpir|124", "Meting|143", "Bholari|164",
        "Kotri Junction|174", "Hyderabad Junction|183", "Detha|190", "Allahdino Sand|205",
        "Palijani|213", "Oderolal|221", "Wahab Shah|228", "Tando Adam Junction|237",
        "Jalal Marri|246", "Shahdadpur|256", "Lundo|270", "Sarhari|280",
        "Nawabshah Junction|298", "Bucheri|311", "Daur|323", "Bandhi|337", "Kot Lalloo|348",
        "Pad Idan Junction|358", "Bhiria Road|371", "Lakha Road|384", "Mehrabpur Junction|398",
        "Setharja|411", "Ranipur Riyasat|420", "Gambat|427", "Tando Mustikhan|442",
        "Khairpur|456", "Begmanji|467", "Rohri Junction|481", "Mando Dairo|489", "Sangi|501",
        "Pano Akil|513", "Mahesar|525", "Ghotki|539", "Sarhad|550", "Mirpur Mathelo|564",
        "Daharki|578", "Reti|596", "Shaheed Haider Ali|614", "Machi Goth|624",
        "Sadikabad|632", "Adam Sahaba|643", "Rahim Yar Khan|654", "Tarinda|666",
        "Kot Samaba|675", "Sahja|685", "Khanpur Junction|696", "Jetha Bhutta|706",
        "Firoza|718", "Liaquat Pur|741", "Chanigot|760", "Kulab|771", "Dera Nawab Sahib|783",
        "Mubarakpur|798", "Kalanchwala|807", "Samasata Junction|819", "Bahawalpur|831",
        "Adamwahan|838", "Lodhran Junction|847", "Shahidanwala|857", "Rukanpur|863",
        "Dunyapur|878", "Kutabpur|889", "Jahania|905", "Jangal Mariala|922",
        "Mehar Shah|929", "Khanewal Junction|938", "Dera Taj|942", "Rajput Nagar|953",
        "Kacha Khuh|959", "Mohsinwal|969", "Mian Channun|981", "Kassowal|999",
        "Chichawatni|1015", "Harappa|1036", "Sahiwal|1056", "Yousafwala|1066",
        "Okara Cantt|1081", "Okara|1093", "Kissan|1102", "Renala Khurd|1110",
        "Habibabad|1126", "Sehjowal|1133", "Pattoki|1139", "Changa Manga|1152",
        "Bhoe Asal|1160", "Kot Radha Kishan|1168", "Prem Nagar|1175", "Raiwind Junction|1183",
        "Jia Bagga|1192", "Kana Kacha|1201", "Kot Lakhpat|1208", "Walton|1212",
        "Lahore Cantt|1218", "Lahore Junction|1223", "Badami Bagh|1225",
        "Shahdara Bagh Junction|1230", "Kala Shah Kaku|1240", "Muridke|1249",
        "Sadhoke|1259", "Kamoke|1269", "Eminabad|1278", "Gujranwala|1291", "Gujranwala Cantt|1299", "Ghakkhar Mandi|1306", "Dhaunkal|1315",
        "Wazirabad Junction|1322", "Haripur Band|1325", "Gujrat|1336", "Deona Juliani|1346",
        "Lala Musa Junction|1355", "Chak Pirana|1362", "Kharian Cantt|1365", "Kharian|1371",
        "Choa Kariala|1381", "Jhelum|1389", "Kala Gujran|1394", "Kaluwal|1401", "Dina|1407",
        "Ratial|1413", "Domeli|1420", "Bakrala|1426", "Tarki|1431", "Sohawa|1439",
        "Missa Keswal|1449", "Gujar Khan|1458", "Ghungrila|1465", "Mandra Junction|1472",
        "Kaliamawan|1481", "Mankiala|1486", "Sihala|1496", "Chaklala|1507",
        "Rawalpindi|1512", "Nur|1515", "Madina-Tul-Hijjaj|1522",
        "Golra Sharif Junction|1527", "Sangjani|1537", "Taxila|1544",
        "Wah Cantt|1547", "Budho|1552", "Wah|1556", "Hasan Abdal|1560", "Burhan|1570",
        "Faqirabad|1579", "Sanjwal|1587", "Attock City Junction|1595", "Rumian|1605",
        "Attock Khurd|1612", "Khairabad Kund|1616", "Jahangira Road|1624",
        "Akora Khattak|1630", "Hayat Sher Pao Shaheed|1638", "Nowshera Junction|1643",
        "Khushhal Kot|1650", "Pir Piai|1653", "Pabbi|1664", "Taru Jabba|1669",
        "Nasarpur|1674", "Peshawar City|1682", "Peshawar Cantonment|1687"], chain=True,
      name_en="Main Line 1 (ML-1)"),
    # Kiamari - Karachi City (ML-1's first 5 km, to the port): no passenger train, and neither
    # OSM nor Wikidata places Kiamari station; left out.
    L("lodhran-multan-khanewal", "Lodhran–Khanewal (via Multan)", [
        "Lodhran Junction|0", "Shah Nal|11", "Gilawala|25", "Zarif Shaheed|36",
        "Shujabad|48", "Chak|56", "Sher Shah Junction|72", "Muzaffarabad|78",
        "Multan Cantt|87", "Piran Ghaib|98", "Tatipur|108", "Riazabad|115",
        "Kot Abbas Shaheed|120", "Shamkote|127", "Khanewal Junction|136"], chain=True),

    # ------------------------------------------------------------------ ML-2
    # Kotri - Attock (no km posts in its RDT, which numbers stations 1-102). Habib Kot -
    # Jacobabad is ML-3's (km posts there). Cut where service stops: Jacobabad - Kashmor - Dera
    # Ghazi Khan had only the Khushhal Khan Khattak Express, suspended since May 2026 (fuel
    # costs); Dera Ghazi Khan - Kot Adu has the DGK Shuttle; Kot Adu - Attock the Thal Express,
    # the Attock Passenger and the Jand Passenger; Kotri - Habib Kot the Mohenjo Daro Express
    # (back since 21 September 2026).
    L("kotri-habibkot", "Kotri–Habib Kot", [
        "Kotri Junction", "Jamshoro", "Sehwan Sharif", "Dadu", "Mohenjo-daro", "Larkana",
        "Ruk", "Habib Kot Junction"], name_en="Kotri–Attock Line (ML-2): Kotri–Habib Kot"),
    L("jacobabad-dgkhan", "Jacobabad–Dera Ghazi Khan", [
        "Jacobabad Junction", "Kandkot", "Kashmor Junction", "Rojhan", "Mithan Kot",
        "Rajanpur", "Jampur", "Dera Ghazi Khan"], suspended=True,
      name_en="Kotri–Attock Line (ML-2): Jacobabad–Dera Ghazi Khan",
      note="the Khushhal Khan Khattak Express only; suspended since May 2026"),
    L("dgkhan-kotadu", "Dera Ghazi Khan–Kot Adu", [
        "Dera Ghazi Khan", "Kot Adu Junction"],
      name_en="Kotri–Attock Line (ML-2): Dera Ghazi Khan–Kot Adu", note="the DGK Shuttle"),
    L("kotadu-attock", "Kot Adu–Attock City", [
        "Kot Adu Junction", "Leiah", "Karor", "Bhakkar", "Darya Khan", "Kallur Kot",
        "Kundian Junction", "Mianwali", "Daud Khel Junction", "Makhad Road", "Jand Junction",
        "Basal Junction", "Attock City Junction"],
      name_en="Kotri–Attock Line (ML-2): Kot Adu–Attock City"),

    # ------------------------------------------------------------------ ML-3
    # Rohri - Chaman, km posts from Rohri. The Jaffar Express runs Rohri - Quetta daily, with
    # security suspensions of days at a time (pk_sources.md); Quetta - Chaman had only the
    # Chaman Passenger, suspended since May 2026.
    L("rohri-quetta", "Rohri–Quetta", [
        "Rohri Junction|0", "Sukkur|5", "Arian Road|11", "Gosarji|18", "Habib Kot Junction|33",
        "Shikarpur|43", "Sultankot|57", "Abad|72", "Jacobabad Junction|84",
        "Dera Allahyar|97", "Mangoli|101", "Dera Murad Jamali|121", "Nuttall|148",
        "Bakhtiarabad Domki|175", "Damboli|187", "Dingra|210", "Perak|231", "Sibi|243",
        "Nari Bank|251", "Mushkaf|259", "Pehro Kunri|268", "Panir|283", "Peshi|298",
        "Ab-I-Gum|306", "Mach|318", "Hirok|331", "Dozan|336", "Kolpur|343",
        "Spezand Junction|359", "Sar-I-Ab|374", "Quetta|384"], chain=True,
      name_en="Rohri–Chaman Line (ML-3): Rohri–Quetta"),
    L("quetta-chaman", "Quetta–Chaman", [
        "Quetta|384", "Sheikh Mandah|391", "Beleli|396", "Kuchlak|406", "Bostan Junction|417",
        "Yaru|429", "Gulistan|466", "Kila Abdulla|479", "Shelabagh|496", "Sanzala|507",
        "Chaman|526"], suspended=True, chain=True,
      name_en="Rohri–Chaman Line (ML-3): Quetta–Chaman",
      note="the Chaman Passenger only; suspended since May 2026"),

    # ------------------------------------------------------------------ ML-4
    # Spezand - Koh-i-Taftan - Zahedan, km posts from Quetta (Spezand 25). No passenger train
    # since February 2020 (the Taftan Express and the Zahedan Mixed). Track to the border.
    L("spezand-taftan", "Spezand–Koh-i-Taftan", [
        "Spezand Junction|25", "Wali Khan|48", "Kanak|60", "Sheikh Wasil|72", "Galangur|122",
        "Kishingi|147", "Nushki|158", "Ahmedwal|179", "Padag Road|263", "Dalbandin|343",
        "Yakmach|401", "Nok Kundi|513", "Taftan|637", "#XIRPK1"],
      suspended=True, name_en="Quetta–Taftan Line (ML-4)",
      note="no passenger train since February 2020"),

    # ------------------------------------------------------------------ branches with km
    L("shershah-kotadu", "Sher Shah–Kot Adu", [
        "Sher Shah Junction|0", "Chenab West Bank|8", "Muzaffargarh|15", "Budh|30",
        "Mahmud Kot|42", "Gurmani|50", "Sanawan|57", "Kot Adu Junction|72"], chain=True,
      note="the Thal Express and the DGK Shuttle"),
    L("khanewal-wazirabad", "Khanewal–Wazirabad", [
        "Khanewal Junction|0", "Mian Shamir|9", "Makhdumpur Pahoran|19",
        "Jan Muhammad Wala|27", "Abdul Hakim|35", "Darkhana|47", "Jarala|55",
        "Shorkot Cantonment Junction|63", "Chutiana|76", "Dabanawala|84",
        "Toba Tek Singh|93", "Janiwala|106", "Gojra|120", "Kot Abadan|131", "Pakka Anna|136",
        "Sar Shamir Road|145", "Risalewala|161", "Samanabad|165", "Faisalabad|170",
        "Nishatabad|175", "Chak Jhumra Junction|190", "Sahianwala|198", "Dar ul Ihsan|204",
        "Sangla Hill Junction|214", "Marh Balochan|224", "Sukheke|235", "Nautheh|243",
        "Kaleke|250", "Hafizabad|264", "Gajargola|279", "Alipur Chatta|288",
        "Mancher Chatta|294", "Jamke Chatta|301", "Mansurwali|311",
        "Wazirabad Junction|325"], chain=True),
    L("shorkot-lalamusa", "Shorkot–Lala Musa", [
        "Shorkot Cantonment Junction|0", "Khanora|11", "Waryam|21", "Rustam Sargana|30",
        "Muddoki|43", "Jhang Sadar|55", "Jhang City|60", "Thatta Mahla|67", "Chund|80",
        "Shah Jewana|94", "Shah Nikdur|105", "Sobhaga|114", "Haryanwala|123",
        "Sillanwali|133", "Shahinabad Junction|147", "Pindi Rasul|153", "Charnali|158",
        "Sargodha Junction|167", "Mitha Lak|178", "Ajnala|187", "Bhalwal|197",
        "Wil Sonpur|203", "Phularwan|212", "Ratto Kala|217", "Mona|220", "Pind Mukko|224",
        "Pakhowal|228", "Chak Saida|236", "Malakwal Junction|241", "Hariah|251", "Ala|258",
        "Mandi Bahauddin|268", "Chillianwala|281", "Chak Sher Muhammad|285", "Dinga|290",
        "Jaurah Karnana|302", "Akhtar Karnana|303", "Lala Musa Junction"], chain=True,
      note="the RDT's Lala Musa post, 325, is Khanewal - Wazirabad's end figure: OSM's track "
           "has 9 km from Akhtar Karnana (303), not 22, and the line 313 km. The last section "
           "is measured on the track"),

    # ------------------------------------------------------------------ branches, traced km
    L("chakjhumra-shahinabad", "Chak Jhumra–Shahinabad", [
        "Chak Jhumra Junction", "Burj", "Chiniot", "Chenab Nagar", "Lalian",
        "Shahinabad Junction"], name_en="Sangla Hill–Kundian Line: Chak Jhumra–Shahinabad",
      note="the Millat Express and the Mianwali Express"),
    L("sargodha-kundian", "Sargodha–Kundian", [
        "Sargodha Junction", "Wegowal", "Shahpur Sadar", "Shahpur City", "Khushab Junction",
        "Jauharabad", "Hadali", "Mitha Tiwana", "Qaidabad", "Wanbhachran",
        "Kundian Junction"], name_en="Sangla Hill–Kundian Line: Sargodha–Kundian",
      note="the Mianwali Express"),
    L("daudkhel-mariindus", "Daud Khel–Mari Indus", ["Daud Khel Junction", "Mari Indus"],
      note="the Mianwali Express and the Attock Passenger"),
    L("shahdara-sanglahill", "Shahdara Bagh–Sangla Hill", [
        "Shahdara Bagh Junction", "Missan Kalar", "Qila Sattar Shah", "Chichoki Mallian",
        "Sheikhupura", "Farooq Abad", "Sachcha Sauda", "Bahalike",
        "Safdarabad", "Moman", "Sangla Hill Junction"], suspended=True,
      note="the Qila Sattar Shah - Missan Kalar bridge fell in the July 2026 floods (30 of 110 "
           "piers); the Badar and Ghouri Expresses suspended, the Mianwali Express sent via "
           "Lala Musa, no reopening found by October 2026"),
    L("shorkot-sheikhupura", "Shorkot–Sheikhupura", [
        "Shorkot Cantonment Junction", "Pir Mahal", "Kanjwani", "Rahme Shah",
        "Tandliawala", "Jaranwala", "Buchiana", "Nankana Sahib", "Warburton", "Bahuman",
        "Sheikhupura"], suspended=True,
      note="the Ravi Express suspended since May 2026; the Waris Shah Passenger ran to Lahore "
           "over the Qila Sattar Shah bridge, down since July 2026"),
    L("lodhran-raiwind", "Lodhran–Raiwind", [
        "Lodhran Junction", "Dhanote", "Kahror Pakka", "Mailsi", "Vehari", "Mandi Burewala",
        "Arif Wala", "Pakpattan", "Haveli Lakha", "Basirpur", "Mandi Ahmadabad",
        "Kanganpur", "Usmanwala", "Khudian Khas", "Kasur Junction", "Raja Jang",
        "Raiwind Junction"], note="the Fareed Express; the Bulleh Shah Passenger Lahore - "
                                  "Pakpattan since July 2026"),
    L("shahdara-narowal", "Shahdara Bagh–Narowal", [
        "Shahdara Bagh Junction", "Kot Mul Chand", "Narang", "Mehta Suja", "Baddomalhi",
        "Raya Khas", "Pejowali", "Narowal Junction"],
      name_en="Shahdara Bagh–Chak Amru Line: Shahdara Bagh–Narowal",
      note="the Allama Iqbal Express, the Faiz Ahmed Faiz and Narowal Passengers"),
    # Narowal - Chak Amru: no passenger train found, and OSM's track has gaps (Narowal -
    # Jassar, Jassar - Shakargarh): left out.
    L("wazirabad-narowal", "Wazirabad–Narowal", [
        "Wazirabad Junction", "Sambrial", "Sialkot Junction", "Gunna Kalan", "Chawinda",
        "Pasrur", "Qila Sobha Singh", "Narowal Junction"],
      note="the Shaheen Passenger, the Lasani and Sialkot Expresses (Wazirabad - Sialkot); "
           "the Allama Iqbal Express (Sialkot - Narowal)"),
    L("lahore-wagah", "Lahore–Wagah", ["Lahore Junction", "Mughalpura", "Wagah"],
      suspended=True, note="the Samjhauta Express, the only train, ended in August 2019"),
    L("golra-basal", "Golra Sharif–Basal", [
        "Golra Sharif Junction", "Tarnol", "Fateh Jang", "Basal Junction"],
      name_en="Golra Sharif–Basal Line", note="the Kohat Express and the Thal Express"),
    L("jand-kohat", "Jand–Kohat", ["Jand Junction", "Khushalgarh", "Kohat Cantt"],
      name_en="Jand–Thal Line: Jand–Kohat", note="the Kohat Express"),
    L("taxila-havelian", "Taxila–Havelian", [
        "Taxila", "Haripur", "Havelian"],
      note="the Hazara Express and the Rawalpindi Passenger"),
    L("malakwal-pdkhan", "Malakwal–Pind Dadan Khan", [
        "Malakwal Junction", "Haranpur", "Pind Dadan Khan"],
      name_en="Malakwal–Khushab Line: Malakwal–Pind Dadan Khan",
      note="the Pind Dadan Khan Shuttle"),
    L("hyderabad-mirpurkhas", "Hyderabad–Mirpur Khas", [
        "Hyderabad Junction", "Tando Jam", "Tando Allahyar", "Mirpur Khas"],
      name_en="Hyderabad–Khokhrapar Line: Hyderabad–Mirpur Khas",
      note="the Shah Latif Express (the Mehran and Saman Sarkar suspended since May 2026)"),
    L("mirpurkhas-zeropoint", "Mirpur Khas–Zero Point", [
        "Mirpur Khas", "Pithoro", "Chhor", "Khokhrapar", "Zero Point Khokhrapar"],
      suspended=True, name_en="Hyderabad–Khokhrapar Line: Mirpur Khas–Zero Point",
      note="the Marvi Express only, suspended since May 2026; Thar Express ended 2019"),
    # Hyderabad - Badin (the Badin Express ended in 2020) and Bahawalnagar - Fort Abbas: OSM
    # has no track for them; left out.
    L("samasata-minchinabad", "Samasata–Minchinabad", [
        "Samasata Junction", "Hasilpur", "Chishtian Sharif", "Bahawalnagar Junction",
        "Minchinabad"], suspended=True, name_en="Samasata–Amruka Line: Samasata–Minchinabad",
      note="no passenger train since the 2000s; OSM's track ends short of Amruka"),
]
