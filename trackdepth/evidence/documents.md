# Documents that may hold profile drawings or track elevations

Compiled 2026-11-01. "Opened" means I read the text myself. "Profile confirmed" means the text has a figure caption or a reference naming a profile or section drawing. I did not look at any drawing as an image (the text extracts do not carry the pictures), so every "profile confirmed" entry still needs a look at the figure itself.

All figures and sentences that I took from these documents are in `depths.csv` (the quote column).

## 1. Opened, and a profile drawing is confirmed

### The New York Subway: Its Construction and Equipment (1904)
- Publisher: Interborough Rapid Transit Company. Project Gutenberg #17569.
- URL: https://www.gutenberg.org/ebooks/17569 (the text file is https://www.gutenberg.org/cache/epub/17569/pg17569.txt)
- Covers: the original IRT, City Hall to Bronx Park and to Kingsbridge, the Harlem River tunnel, the Brooklyn Extension (Joralemon Street tubes).
- Opened: yes (full text).
- Profile: three plate captions in the text: "Profile of Rapid Transit Railroad Manhattan and ..." (near the station list), "Profile of Brooklyn Extension", "Profile of Harlem River Tunnel and Approaches". The drawings are images on the Gutenberg HTML edition; not viewed.
- Also has depths in text (platforms 90 to 100 ft at 168th, 181st and Mott Avenue; Joralemon tubes 94 ft below mean high water).

### The New York Rapid Transit Railway Extensions (Engineering News, 1914), 13 chapters
- nycsubway.org index: https://nycsubway.org/wiki/The_New_York_Rapid_Transit_Railway_Extensions_(1914)
- Chapter III (profile): https://nycsubway.org/wiki/Chapter_03._General_Arrangement_for_Construction
- Chapter XI (river tunnels): https://nycsubway.org/wiki/Chapter_11._The_River_Tunnels
- Covers: the Dual Contracts IRT lines (Lexington Avenue Line, Seventh Avenue Line, Brooklyn lines) and the Broadway line.
- Opened: yes (chapters 1, 3, 4, 5, 8, 10, 11, 13).
- Profile: Chapter III, "Fig. 7. Profile of the Lexington Avenue Line" (40th Street to 138th Street). The text also says the original subway profile was in Engineering News, 20 Feb 1902 (not found online).
- Same series, reprinted as a 1915 booklet with an extra section on the elevated lines: "Building the New Rapid Transit System of New York City" (Fred Lavis, with Maurice E. Griest). OCR text, opened: https://distantreader.org/stacks/trust/uc2/uc2_ark+=13960=t44q7sd2w.txt (this is a HathiTrust scan served by Distant Reader, no HathiTrust login needed). A page-image scan would show the figures.

### Jay-Smith Street and Fulton Street Subway (1934)
- Author: Alfred Brahdy, designing engineer, Board of Transportation. Civil Engineering, reprinted on nycsubway.org.
- URL: https://nycsubway.org/wiki/Jay-Smith_Street_and_Fulton_Street_Subway_(1934)
- Covers: IND Jay Street and Smith Street subway, Fulton Street Line to Rockaway Avenue.
- Opened: yes.
- Profile: figure "Plan and profile of the Jay-Smith Street Subway". Also defines the elevation datum used in Brooklyn IND drawings (100 ft below mean high water of New York Bay) and gives street elevations along Fulton Street.

### The Construction of the Harlem River Tubes (1915)
- Author: Howard Babcock Gates, civil engineering thesis, University of Wisconsin. nycsubway.org.
- URL: https://nycsubway.org/wiki/The_Construction_of_the_Harlem_River_Tubes_(1915)
- Covers: Lexington Avenue Line tubes under the Harlem River, with a comparison to the 1904 Lenox Avenue (West Farms branch) crossing.
- Opened: yes.
- Profile: "Figure No. 2 shows in plan and profile the location" of the Lexington Avenue tubes; also a figure "Profile of IRT Contract One Harlem River tunnel" (method used for west half and east half).

### Second Avenue Subway Phase 2 Supplemental Environmental Assessment (MTA, 2018), Chapter 2 "Description of Phase 2 Modified Design"
- URL: https://www.mta.info/document/67346 (46 pages; new.mta.info/document/67346 redirects to it)
- Covers: Second Avenue Subway Phase 2 (96th Street to 125th Street) and how it differs from the 2004 FEIS design.
- Opened: yes, through the fetch tool (it saves the PDF); curl gets 403.
- Profile: Figure 2-7 "SAS Phase 2 Geological Profile" (page 37 of the PDF) showing soil, rock, and where cut-and-cover and mined construction is used. Figures 2-3a (125th Street curve) and 2-4a (125th Street Station) compare the 2004 and modified designs.
- Companion documents I opened: Executive Summary https://www.mta.info/document/67331 (28 pages, has a figure labelled "tunnel approximately 20' deeper") and Chapter 1 https://www.mta.info/document/67341.

### Second Avenue Subway FEIS (MTA/FTA, 2004), Chapter 2 "Project Alternatives"
- URL: https://www.mta.info/document/22101 (27 pages)
- Covers: the full 8.5 mile line, all 16 stations.
- Opened: yes.
- Profile: no figure in the text I extracted, but Table 2-1 gives the depth to base of rails for every station (all entered in `depths.csv`). The chapter sends the reader to Figure 3-19 in Chapter 3, which is "the alignment, stations, and tunnel in relation to street level, bedrock, and other tunnel structures". That is the profile to get.
- Chapter 3 (construction) is https://www.mta.info/document/22106 (63 pages, opened). Figure 3-19 is referred to on page 51 but the picture is not in the extract. Probably in the separate FEIS figures file. I did not find it.
- Chapter 1 is https://www.mta.info/document/22096 (opened, nothing on depth).

## 2. Opened, depth numbers but no profile found

- Subways and Tunnels of New York (Gilbert, Wightman and Saunders, 1912). archive.org item `subwaystunnelsof00gilb`, OCR text https://archive.org/download/subwaystunnelsof00gilb/subwaystunnelsof00gilb_djvu.txt. Methods and costs of the IRT, McAdoo (Hudson) tubes and others. Only the 94 ft Joralemon figure is relevant. No subway profile seen in the text.
- The Steinway Tunnels (David Rogoff, 1960). https://nycsubway.org/wiki/The_Steinway_Tunnels_(1960). Gives depth figures for the Steinway Tunnel and Times Square; the word "profile" appears only in running text.
- Under-River Tunnel Headings Meet (Old Slip to Clark Street, 1916) https://nycsubway.org/wiki/Under-River_Tunnel_Headings_Meet_(1916) and New Clark Street Tunnel Completed (1919) https://nycsubway.org/wiki/New_Clark_Street_Tunnel_Completed..._(1919). One depth figure each (88 ft and 85 ft).
- Brooklyn Approaches to Downtown Tubes (Aksel H. Jorgensen, 1917). https://nycsubway.org/wiki/Brooklyn_Approaches_to_Downtown_Tubes_(1917). Shaft and cut depths; there is a cross section of the Borough Hall approach but no profile.
- Design of the Diagonal Station and Connection (Olof A. Nilsson, Public Service Record, Dec 1918). https://nycsubway.org/wiki/Design_of_the_Diagonal_Station_and_Connection_(1918). Grand Central, with plates 1 to 7 (plan and cross sections); gives a mezzanine elevation of 128.4 ft.
- Rapid Transit in New York City and in the Other Great Cities (1906), Chapter 14. https://nycsubway.org/wiki/Chapter_14:_Engineering_Features_of_the_New_York_Subway. Shaft depths for the deep IRT stations.
- Nothing usable on depth in: The Dual System of Rapid Transit (1912), Fourth Avenue Subway in Brooklyn (1915), Two Lines of the Dual System in Operation (1915), Broadway Subway Now Open (1917), Fifty Years of Rapid Transit (1918), History of the Independent Subway, IND Second System 1929 Plan, The Design of Subways (1918; only grade rules such as a 0.75 percent limit at stations and 3 percent between stations), all on nycsubway.org.
- LPC archaeological report for Second Avenue Subway 96th Street Station (AKRF, 2009), https://s-media.nyc.gov/agencies/lpc/arch_reports/1185.pdf. Opened. Street elevations over time and fill and bedrock depths along Second Avenue. Useful for ground and bedrock (the other session), not for the tunnel.
- Wikipedia article text: all station and tunnel articles in the four boroughs' station categories plus the Staten Island Railway were read. See `depths.csv`.

## 3. Not opened, or could not be opened (Anita to open)

Blocked (403) or too big for the fetch tool:
- FTA project management oversight reports and technical memos for Second Avenue Subway. transit.dot.gov returns 403 to curl and to the fetch tool.
  - https://www.transit.dot.gov/sites/fta.dot.gov/files/docs/foia/65176/pmoc-report-secondavesubway-april-2010.pdf
  - https://www.transit.dot.gov/sites/fta.dot.gov/files/TechMemo7_2010Dec16.pdf
  - https://www.transit.dot.gov/sites/fta.dot.gov/files/docs/TechMemo5_2010Apri27Memo.pdf
  These should list station depths as built or as designed.
- MTA FEIS landing page https://www.mta.info/node/9171 (403). It lists every chapter and the figure files. The chapter PDFs themselves open through https://www.mta.info/document/<number> in the fetch tool. Chapters 1, 2 and 3 are 22096, 22101 and 22106 (22101 I guessed from the sequence). Document 22131 (2 pages) and 22281 (7 pages, FEIS figures 10-1 to 10-6 for the historic resources agreement) are not about depth. Other chapters and the figure volumes are probably nearby numbers; I did not try them.
- MTA document https://www.mta.info/document/67446 (found by search; larger than 10 MB, the fetch tool refused it). Probably the figures for the Phase 2 SEA.
- HathiTrust full-text search and page views (403 in an earlier session). Distant Reader serves some HathiTrust OCR text directly (see above), which might get around it for public domain volumes.

Found by search, not opened:
- Report of the Public Service Commission for the First District of the State of New York (annual reports to 1919; 1916 volume at Columbia): https://dlc.library.columbia.edu/catalog/cul:4tmpg4f4rq . These carry the contract drawings summaries and tunnel profiles for the Dual Contracts. Also NYC government publications portal items (titles unknown): https://a860-gpp.nyc.gov/downloads/dz010v78j , https://a860-gpp.nyc.gov/downloads/xk81jr150 , https://a860-gpp.nyc.gov/downloads/zk51vn52s .
- The New York Rapid-transit Subway (digitised page images, Dansk Teknisk Kulturarv): https://tekniskkulturarv.dk/book/394431e0-e6b8-405a-a100-32063b0e6179?page=51 . Page images, probably the 1904 IRT book with its plates, so the profile drawings may be viewable there.
- Geotechnical Considerations, No. 7 Subway Line Extension Project in New York City (paper, paywalled abstract): https://onemine.org/documents/geotechnical-considerations-no-7-subway-line-extension-project-in-new-york-city
- ASCE paper on the 72nd Street Second Avenue Subway cavern station (listed in the Wikipedia article for the station): https://ascelibrary.com/doi/10.1061/9780784479742.033
- Datum conversions between Brooklyn Highway Datum, NGVD 1929 and NAVD88 appeared in search snippets from geotechnical reports (one says NAVD88 elevation 0 is 1.447 ft below Brooklyn Highway Datum). I did not open the source, so the numbers are not in `depths.csv`. A candidate document: https://www.geiconsultants.com/wp-content/uploads/2020/06/Attachment-A-3-Appendix-B-30-Field-Activity-Reports_Section-1.pdf
- NYC Museum subway construction photograph collection, IND routes 101 (Cranberry Street tunnel) and 103 (Rutgers Street tunnel): https://digitalcollections2.nyhistory.org/items/942-subway-construction-photograph-collection-independent-city-lines-route-101-1926-1944 . Photos, no numbers expected.
- Engineering News, 20 Feb 1902: profile of the original subway (named in Chapter III of the 1914 series). Not found online.

## 4. Downloaded by hand into raw/docs/ (read 2026-11-02)

Points read off these drawings are in `profiles_docs.csv` (same columns as `profiles.csv`; build.py reads every `profiles*.csv`).

### Second Avenue Subway FEIS 2004, Figure 3-19 (`figure3-19.pdf`)
- What it is: one page, "Construction Techniques": a plan of the whole line with a profile strip under it (stations, tunnel coloured by construction method, soil and rock, existing subways). An inset at the right shows 125th Street. Dated 3.11.04, "Map for conceptual purposes only".
- No vertical scale, no datum, no grid. The strip is a single raster image (3778 x 1507 px). I fitted a scale on the 16 stations' Table 2-1 depths to base of rails (chapter 2, already in `depths.csv`): depth = 3.12 ft per px of (box bottom minus drawn ground) - 16.1 ft, residual sd 3.7 ft over 15 stations (125 St is in the inset, not used). The strip is roughly true to scale (about 2.6-3.1 ft per px vertically, 4.4 m per px along the line).
- Used: 25 points on the built part (IND Second Avenue Line, 67 St to 103 St), every ~300 m between stations, rail = DEM street minus the drawn depth. Station centres were left out (they are the Table 2-1 numbers, and build.py prefers as-built figures at 72, 86 and 96 St). South of 67 St the drawn line is the unbuilt main line, not the 63 St connector, so nothing is placed there.
- Caveats: it is the 2004 design. The built stations at 72, 86 and 96 St came out 1-4 m deeper, so the points between 67 and 97 St carry sd 3.6-4.9 m. From 98 to 103 St the line runs in the 1970s tunnel segment (99-105 St), which is drawn just under the street (rail about 18-26 ft down); chapter 3 calls these segments "close to the street surface".
- Not used: everything south of 63 St and north of 105 St (no OSM track); the 125 St inset (no scale, unbuilt).

### Second Avenue Subway FEIS 2004, Chapter 3 (`chapter03.pdf`, 63 pages)
- Text and Table 3-1 (construction method by block, north to south). It names Figure 3-19 on p. 3-51 but carries no depth table.
- Useful sentences: the 63rd Street connector tunnels are "in the rock approximately 60 to 80 feet beneath private property near 63rd Street" (p. 3-39); Phase 1 uses the 1970s tunnel 99-105 St and connects to it between 98th and 99th Streets (p. 3-39, Table 3-1); the 1970s segments "are close to the street surface" (p. 3-16); at 34th Street "the tunnel elevation needs to be shallow to avoid the existing Amtrak tunnels" (p. 3-23); Hanover Square tail tracks "at a depth of approximately 110 feet along Water Street" (p. 3-60, unbuilt).
- No profile drawing in this file.

### Second Avenue Subway Phase 2 Supplemental EA, 2018, full document (`SAS_Phase_2_Supplemental_EA-07-2018.pdf`, 230 pages)
- Profiles: Figure 2-7 "Construction Overview" (PDF p. 84; the chapter-only file `sas2018/67346.pdf` has the same drawing on p. 37) is a schematic of 106 St to 125 St with soil and rock and no vertical scale. Figure 2-6 (p. 74) is a generic ancillary-facility section. Figures 2-1a to 2-5a and S-3 to S-7 are plans only (the 2-3a plan has the "tunnel approximately 20' deeper" arrow).
- Text: only relative depth (125 St curve and station "about 20 feet deeper" than the 2004 design, already TD0046-48), the 106th and 116th St stations "relatively shallow to align with the existing tunnel segments that were constructed during the 1970s" (p. 2-16), and flood elevations in NAVD88 (100-year BFE 12.0 ft; design flood elevation 17.9 ft NAVD88).
- Nothing used: no scale and no track there in OSM (Phase 2 is unbuilt). `profiles.csv` has no rows from the earlier chapter file either.

### Hudson Tunnel Project FEIS, May 2021, Chapter 2 (`02 Alternatives and Preferred Alternative_2021-05-27.pdf`, 55 pages)
- New tunnel (Gateway, under construction, two tubes south of the existing ones): Figure 2-5 (NJ, plan and profile, elevation in feet), Figure 2-6 (Hudson River and Manhattan, plan and profile, 100 to -350 ft) and Figure 2-7 (Manhattan, 12th Av to 10th Av, with a "Top of Rail" line, the existing grade, top of rock and the No. 7 line tubes). These are vector drawings, read exactly from the PDF paths. Datum not named; taken as NAVD88 (the river is drawn at 0, and the drawn grade, +5 ft at 12th Av and +20 ft at 10th Av, matches the DEM's 1.8 m and 5.3 m).
- New-tunnel top of rail: about -134 ft at its low point under the river, -61 ft at the Manhattan bulkhead, -58 ft at 12th Av, -39 ft at 11th Av, -20 ft at 10th Av, rising to about -16 ft in A Yard east of Dyer Av, where it meets the existing Penn Station approach tracks (A Yard tracks "must be lowered to meet the alignment"). Grade at most 2.1%. Text: crown 25-50 ft below the river bottom; "about 45 feet beneath Hudson River Park and Twelfth Avenue"; "approximately 35 feet below the surface at the shaft site"; "approximately 50 feet above the No. 7 subway line, which runs under Eleventh Avenue".
- Existing North River Tunnel (1910, Amtrak/NJ TRANSIT into Penn Station): no profile and no elevation in this chapter. Figure 2-14 is a cross section only (bench walls 5 ft 10 in above top of rail). The text only says the new tunnel runs "at a similar vertical elevation as the existing North River Tunnel" under the Palisades. The "Existing A Yard" callout on Figure 2-7 marks where the new tunnel ends, not an elevation of A Yard.
- Used: 9 Manhattan points on the new tunnel (none land: no OSM track yet, build.py skips them; they are there for when it opens) and 1 point on the No. 7 tail tracks under 11th Av (-30.3 m; those OSM tracks are service=yard, so build.py skips it too; the model has -31 to -35 m there).
- Not used: the river and New Jersey part of the new tunnel (positions under the river could only be guessed from the plan).

### FTA technical memos on Second Avenue Subway, 2010 (`TechMemo5_2010Apri27Memo.pdf`, `TechMemo7_2010Dec16.pdf`)
- Memo 5: entrance changes at 63rd St and Third Av (a stair moved 7 ft). Memo 7: size and look of the 72nd St ancillary facility. No depths, no profiles. Nothing used.

Searched for and not found online: any paper on the IND East River tunnels (Cranberry Street, Rutgers Street, 53rd Street, 63rd Street) with a profile; the Civil Engineering and Engineering News-Record 1930s volumes are the likely place. The 14th Street (Canarsie) tunnel, Montague Street tunnel and Whitehall Street tunnel depths were also not found.
