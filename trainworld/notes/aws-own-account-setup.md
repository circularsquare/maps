# Your own AWS account: setup

About 20-30 minutes. Once you're done, an agent sets up the benchmark machine; you don't need to do anything in AWS after this.
Keep it separate from your friend's account: a different email, its own CLI profile name.

## 1. Make the account
1. Go to https://aws.amazon.com and click Create an AWS account. Use an email you don't use for the friend's account.
2. Account name: anything, e.g. "anita-bench".
3. When it asks which plan: pick the **Free plan**. As of mid-2025 that came with $100 of credits at sign-up and up to
   $100 more for trying a few services, for 6 months; check what the page says now. The Free plan can't run up a bill:
   when the credits run out it stops things instead of charging. (The paid plan keeps the credits but bills past them.)
4. It asks for a card and a phone check either way.

## 2. Lock the root login
1. Sign in as root (the email you used). Top right, your account name, then Security credentials.
2. Under Multi-factor authentication, Assign MFA device: an authenticator app is easiest.
3. Don't make access keys for root.

## 3. A day-to-day user for the CLI
1. In the search bar, open **IAM**. Users, Create user. Name: `bench`. Don't give it console access.
2. Permissions: Attach policies directly, pick **AmazonEC2FullAccess** and **AmazonS3FullAccess**. (An agent can tighten
   these later.)
3. Open the new user, Security credentials tab, Create access key, choose "Command Line Interface (CLI)", tick the box,
   create. Keep that page open for step 5. Never paste the keys into chat.

## 4. A spending alarm
1. Search **Billing and Cost Management**, then Budgets, Create budget, "Zero spend budget" or a monthly cost budget of $10.
2. Put your email in. You'll get a mail the moment anything costs money.

## 5. The CLI on this PC
1. If `aws --version` doesn't work in PowerShell, install it: https://awscli.amazonaws.com/AWSCLIV2.msi
2. In PowerShell: `aws configure --profile anita-own`
   - Access key and secret: from step 3.
   - Region: `us-east-2` (Ohio, same as the friend's account, cheap and near).
   - Output: `json`
3. Check: `aws sts get-caller-identity --profile anita-own` should print an account number that is **not** the friend's.

## 6. Ask for enough cores
New accounts can often only run a few vCPUs at once, too few for a 16-core bench machine.
1. Search **Service Quotas**, AWS services, Amazon EC2, "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances".
2. Request a quota increase to **32**. It's usually approved within a day; you get an email.

## Then tell me
"aws account ready, profile anita-own". An agent then:
- adds the profile to trainworld's AWS rules (docs/aws.md) so it's never mixed up with the friend's account,
- builds a machine image with Node, Chrome, Rust and the game's data,
- writes one script that starts the machine, runs the edit bench / perf bench / smoke on a given commit, copies the
  results back and shuts it down, so nothing is left running.
