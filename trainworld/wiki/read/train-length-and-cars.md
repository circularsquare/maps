# Train length and cars

Each line has a train length in cars. A new line starts at the length you set last, or 5 cars before you have set one. A crowded line is the cue to lengthen it.

**What a car holds.** A car is 20 m long and has 44 seats. At its rated load it carries 160 people, and squeezed to the limit 240 ([full trains](full-trains.md)). A longer train holds that many more.

**The platform sets the most.** The longest setting runs as many cars as the shortest platform on the line fits. If a platform is shortened below the length you set, the trains shorten with it, and lengthening the platform brings your length back.

**Cars cost money once.** A car costs US$2.5M, a high-speed car US$3.75M. The network owns what its busiest timetable needs: trains times cars. A change that needs more cars, such as longer trains or a shorter headway, buys the difference and is refused if the cash won't cover it. Cars are never sold. Those a line stops needing wait in the depot for the next line that grows. High-speed cars and ordinary cars are kept apart, so spares of one kind don't serve the other.

**Cars cost money every kilometre.** A train costs US$4 a km plus US$2 a km for each car, so a four-car train runs at US$12 a km. High-speed trains cost US$5 plus US$3 a car. Running costs are mostly wages, so a poorer country pays less: the US figure times the square root of the country's income per head over the US's ([upkeep](upkeep.md) is the cost of the track itself).

Exact rule: `docs/track-and-service.md` "Train length" and "Trains" under money.
In the code: **main** `sim/src/track/wasm/lines.rs` `set_line_cars`, `sim/src/track/boarding.rs` `RATED_PER_CAR`, `CRUSH_PER_CAR`, `SEATS_PER_CAR`, also `sim/src/track/params.rs` `CAR_LEN`, `CAR_PRICE`, `HS_CAR_PRICE`, `RUN_COST_TRAIN_KM`, `RUN_COST_CAR_KM`, `HS_RUN_COST_TRAIN_KM`, `HS_RUN_COST_CAR_KM` `app/src/workers/saveFile.ts` `NEW_LINE_CARS` `app/src/game/income.ts` `runScale`
