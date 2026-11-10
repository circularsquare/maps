# Zones and realms

The world is cut into zones, and every place is in one zone or in none.

**City zones** are metro areas. **Countryside zones** are the land around them that belongs to no metro: a county, a district, a prefecture, whatever unit the country counts people in. A countryside zone holds at least about half a million people, so thin country is merged into bigger units. Both kinds work the same way. Each has its own trips, its own station-walk areas, a boundary and a place in the long-trip market ([between cities](between-cities.md)). Everything a city has, a countryside zone has too.

**Dead space** is land in no zone: countries the game doesn't cover yet, and very empty ground inside covered ones. No trips start or end there, so a line through it carries only riders passing through.

A zone loads when you look at it or when a station reaches it. Until it has loaded it is drawn grey.

**Realms** are country-sized groups of zones, such as Germany or Japan, with big countries split into several. A realm is shown while you have a station or track in it, while it holds your start, or once you have shown it. To show one, click its shaded land or its circle, or rest the view over it, centred on its land, at zoom 10 or closer. A realm you showed stays shown for the session. Reloading hides it again unless you have built there.

Realms that aren't shown sit under a darker shade, with one circle each. Their zones draw nothing and load nothing, so the map stays quick. Hiding a realm changes no trips and no money. Demand and fares are the same whatever you've shown.

Zoomed out past zoom 2.5, there are no circles and no shade.

Exact rule: `docs/zones.md` "1. What a zone is", "2. How non-city zones are built" and its "Realms" paragraph; `docs/ui-map.md` "Realms not shown" for the shade, circles and clicks. Which countries belong to which realm is a hand list in `app/src/game/realms.json`.
In the code: **main** `app/src/game/realms.ts` `app/src/map/realmOverlay.ts` `AUTO_SHOW_ZOOM`
