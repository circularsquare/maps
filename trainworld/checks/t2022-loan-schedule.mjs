// T-2022 check: a loan pays itself back in equal daily payments over its term (app/src/game/loan.ts),
// drained in uneven slices as the clock worker settles: each whole day pays the same, the total is
// the term times the payment, nothing is left or paid after the last day, a payment goes whatever the
// cash, a save round trip keeps the schedule, and a pre-T-2022 save's loan starts a full term.
//   node --experimental-strip-types trainworld/checks/t2022-loan-schedule.mjs
import { LOAN_AMOUNTS, LOAN_RATE, LOAN_TERM, loanBalance, loanDrain, loanFromSave, loanPayment, loanToSave, newLoan, loanAllowed } from '../app/src/game/loan.ts';

const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

let seed = 2022;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);

for (const amount of LOAN_AMOUNTS) {
  const l = newLoan(amount);
  report(`$${amount / 1000}B: the payments are worth the loan at ${LOAN_RATE * 100}% a day`, near(loanBalance(l), amount), `${loanBalance(l)}`);
  // settle-like slices: each hour in one to four pieces, some days skipped over in one long slice
  const perDay = [];
  let paid = 0, t = 0;
  while (t < (LOAN_TERM + 3) * 24) {
    const pieces = 1 + Math.floor(rand() * 4);
    let used = 0;
    for (let i = 0; i < pieces; i++) {
      const dh = i === pieces - 1 ? 1 - used : (1 - used) * rand();
      used += dh;
      const owed = loanDrain(l, dh / 24);
      const day = Math.floor(t / 24);
      perDay[day] = (perDay[day] ?? 0) + owed;
      paid += owed;
    }
    t += 1;
  }
  const term = perDay.slice(0, LOAN_TERM), after = perDay.slice(LOAN_TERM);
  report(`$${amount / 1000}B: ${LOAN_TERM} equal payments of ${l.pay.toFixed(3)}M`, term.length === LOAN_TERM && term.every((v) => near(v, l.pay, 1e-9)), `min ${Math.min(...term)} max ${Math.max(...term)}`);
  report(`$${amount / 1000}B: total repaid is the term times the payment`, near(paid, l.pay * LOAN_TERM, 1e-9), `${paid} vs ${l.pay * LOAN_TERM}`);
  report(`$${amount / 1000}B: ends at zero on the last day, nothing after`, l.days === 0 && after.every((v) => v === 0) && loanBalance(l) === 0, `days ${l.days}, after ${after.join(',')}`);
}

// the balance a day in follows the bank's sum: last balance plus a day's interest minus the payment
{
  const l = newLoan(5000);
  let b = 5000, ok = true;
  for (let d = 0; d < 10; d++) {
    for (let h = 0; h < 24; h++) loanDrain(l, 1 / 24);
    b = b * (1 + LOAN_RATE) - l.pay;
    ok &&= near(loanBalance(l), b, 1e-9);
  }
  report('balance after 10 days matches interest on what is left', ok, `${loanBalance(l).toFixed(3)} vs ${b.toFixed(3)}`);
}

// a short-cash day: the payment still goes and the cash goes below zero
{
  const l = newLoan(1000);
  let cash = 5;
  for (let h = 0; h < 24; h++) cash -= loanDrain(l, 1 / 24);
  report('short cash: the day\'s payment still goes, cash below zero', near(cash, 5 - l.pay) && cash < 0 && near(l.days, LOAN_TERM - 1), `cash ${cash.toFixed(3)}`);
}

// a save round trip mid-term, through JSON as the save header goes
{
  const l = newLoan(10000);
  loanDrain(l, 17.4);
  const back = loanFromSave(JSON.parse(JSON.stringify(loanToSave(l))));
  report('save round trip keeps amount, payment and days left', back && back.amount === l.amount && back.pay === l.pay && back.days === l.days, JSON.stringify(back));
  const done = { ...l };
  loanDrain(done, LOAN_TERM);
  report('a paid-off loan saves as none', Object.keys(loanToSave(done)).length === 0 && loanFromSave(loanToSave(done)) === null && loanFromSave({}) === null);
  const old = loanFromSave({ loan: 5000 });
  report('a pre-T-2022 save\'s $5B starts a full term', old && old.amount === 5000 && old.days === LOAN_TERM && near(old.pay, loanPayment(5000)), JSON.stringify(old));
}

// T-1906's rule on the payment: $40M a day of profit allows $1B, not $5B
{
  const p = { perDay: 40, days: 7 };
  report('profit rule on the payment: $40M a day allows $1B, refuses $5B', loanAllowed(1000, p) && !loanAllowed(5000, p) && loanAllowed(1000, null) && !loanAllowed(5000, null), LOAN_AMOUNTS.map((a) => `${a}: ${loanPayment(a).toFixed(2)}`).join(', '));
}

console.log(results.every(Boolean) ? 'T-2022 PASSED' : 'T-2022 FAILED');
process.exit(results.every(Boolean) ? 0 : 1);
