# The $20 cutoff on your own AWS account

About 10 minutes, signed in as root (your email and MFA). When your account's spending this month reaches $20, AWS
attaches a policy to the `bench` user that stops it starting machines. A machine already running stops by itself at
its timer (3 hours at most, about 75 cents at the worst).

Two things to know:
- AWS updates its spending figures a few times a day, so the cutoff can come a few hours after the $20 is reached.
- AWS's own "stop machines" option only stops machines named ahead of time, and every bench run makes a new one, so the
  timer does that part.

If a label on screen differs a little from the one here, pick the closest one.

## 1. The policy that blocks new machines

1. In the search bar at the top, type **IAM** and open it.
2. In the left menu, click **Policies**.
3. Click **Create policy**.
4. Click **JSON** (top right of the editor).
5. Delete everything in the editor and paste this:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "NoNewMachinesOrDisks",
      "Effect": "Deny",
      "Action": [
        "ec2:RunInstances",
        "ec2:StartInstances",
        "ec2:RequestSpotInstances",
        "ec2:RequestSpotFleet",
        "ec2:CreateFleet",
        "ec2:CreateVolume",
        "ec2:CreateSnapshot",
        "ec2:CreateSnapshots",
        "ec2:CreateImage",
        "ec2:CopyImage",
        "ec2:CopySnapshot",
        "ec2:AllocateAddress",
        "ec2:CreateNatGateway",
        "ec2:CreateVpcEndpoint",
        "ec2:ModifyInstanceAttribute",
        "s3:CreateBucket"
      ],
      "Resource": "*"
    }
  ]
}
```

6. Click **Next**.
7. Policy name: `bench-cutoff`
8. Click **Create policy**.

It blocks nothing yet: it only takes effect once it is attached to `bench`, which the budget does in part 3. Stopping
machines and deleting things stay allowed, so a run can still clean up after the cutoff.

## 2. The role the budget acts through

1. Still in IAM, in the left menu, click **Roles**.
2. Click **Create role**.
3. Trusted entity type: click **AWS service**.
4. Service or use case: pick **Budgets**.
5. Click **Next**.
6. In the search box, type `AWSBudgetsActionsWithAWSResourceControlAccess`.
7. Tick the box next to it.
8. Click **Next**.
9. Role name: `budgets-cutoff-role`
10. Click **Create role**.

If **Budgets** is not in the list at step 4: pick **Custom trust policy** at step 3 instead, paste the text below over
what is in the box, click **Next**, and carry on from step 6.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Principal": { "Service": "budgets.amazonaws.com" }, "Action": "sts:AssumeRole" }
  ]
}
```

## 3. The cutoff on your $20 budget

1. In the search bar, type **Budgets** and open it (it is part of Billing and Cost Management).
2. Click the name of your $20 budget.
3. Click **Edit**.
4. Click **Next** until the page is called **Configure alerts**.
5. If there is an alert with threshold **100**, **% of budgeted amount**, **Actual**, keep it. If not, click
   **Add an alert threshold** and set those three: **100**, **% of budgeted amount**, **Actual**, with your email.
6. Click **Next**. The page is called **Attach actions**.
7. Click **Add action**.
8. Alert: pick the 100% one.
9. Select IAM role: `budgets-cutoff-role`
10. Which action type: **Identity and Access Management (IAM) policy**
11. Policy: `bench-cutoff`
12. Users: `bench`
13. Do you want to automatically run this action: **Yes**
14. How do you want to be alerted: **Use the same alert settings when you defined this threshold**
15. Click **Next**.
16. Click **Save**.

Back on the Budgets page, the budget's **Actions** column should say 1.

## 4. Optional: a $100 alarm

A second line of defence, only an email.

1. On the Budgets page, click **Create budget**.
2. Pick **Use a template (simplified)**.
3. Pick **Monthly cost budget**.
4. Budget name: `alarm-100`
5. Enter your budgeted amount: `100`
6. Email recipients: your email.
7. Click **Create budget**.

Your first two budgets cost nothing.

## When the cutoff has fired

You get an email, and a bench run says "AWS refused to start a machine". The block stays until it is undone, possibly
into next month. To undo it (a new month, or you raised the budget): Budgets, click the budget's name, in **Alerts**
click the 100% alert, and in **Action history** pick the action and choose **Reverse**.
