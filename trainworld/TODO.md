# trainworld — Anita's todo

## Decide

- [x] No signals for now, with tangles discouraged by capacity and the door left open for signals
      later
- [x] First region: Northeast US, New York first
- [x] One train kind
- [x] Towns can become full cities mid-game
- [x] Rust for the simulation
- [x] Accounts and shared saves: shonei-server, for now
- [x] Name: anitabuilder (front-facing; trainworld stays the folder and code name)

## Do

- [x] Install Rust. Done 2026-10-08, Rust 1.99.
- [ ] Next time you play Subway Builder or NIMBY Rails, note the moments that feel laggy or
      unsatisfying. They become test cases.



come up with game name 
more cities

figure out better work demand model... people should not be commuting from downtown philly to camden

non work demand
find estimates of how much demand is nonwork in various cities
departure times for various cities?
make "add city" checklist?

try to fit to real nyc OD data? look for studies in various countries to fit 

station complexes, and better station rendering: draw stations as boxes along the track rather than circles  (for placement ghost too they should draw as box)
      rework the station circle sizing by demand... have that be a special mode where they draw as much bigger circles with not full opacity (currently they just look small and undistinguishable)
also the track palcement ghost looks weird with how the dashed-ness of the line interacts with curves, we should have it not be dashed, rather like a solid line that is colored dynamically with whatever color mode we have (so like track height if we have height selected, or max speed, or if neither of those is selected we can just have it be white body i guess)

in station panel we should be able to switch between station and station complex view (will aggregate station bubbles on the map too), criteria for being a station complex should be like are the platforms close to touching

if we click on a station complex we should be able to see like how much traffic is on each line and maybe a zoomed in map of all the platforms and draw the rough transfer flows between each of them as lines, would be cool. though for how we made the new york save it'd be like tons of ocnnections cuz we have at like grand central these huge confluences of like 20 lines. so maybe cap it to show the top ones or something. to be clear this mini map should draw on the bottom left panel in the station info. this is maybe low prio

i dont like how in the cities panel we have a list of all the cities... that seems unscalable, right? lets get rid of that list. 

lets put as a low prio task a search bar for finding a place (a station we've placed, or a line name, or a city, or maybe a neighborhood / address)

i think the cities panel should just show like what the currently viewed city is, and what the top intercity demands are from here. with little bars like the bar on the top of screen bar saying how much fo each of those demands are satisfied by train and how much are other (red)

before demand data is loaded, on the demand selector lets fade out homes/destinations (though it should still like preserve selection from before, so like if u have homes selected and then refresh it should show grayed out red homes) and on mouse over say calculating demand

for the demand types breakdown lets show a pie chart with each segment of demand, and also color the individual demand plots. each segment gets a color, maybe work blue, school bluegreen, social purple, shopping red, special events green?

for demand calculation lets do the city we're looking at first, and not run the game until the city we're looking at is fully loaded. we should show a "loading..." with a percent loaded and not allow any interaction until at least one city is loaded. also for cities that dont have any construction inside them we dont need to load these.

for shcool coverage, should search for some semiglobal university enrollment / cordinates database
and also try to find something more general than nyc doe building enrollment (doesnt have to be quite as accurate either, though its nice to have for sure)


be able to filter to just one demand segment and see only that segment on the map? 

when track disply is line + traffic thickness we should show tooltip when mousing over a line segment and holding it there for like >400ms that says the daily traffic. and simialrly for line/height/maxspeed we should show a tooltip that says the info that is shown