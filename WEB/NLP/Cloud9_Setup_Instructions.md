# Setting Up AWS Cloud9 for the NLP Web Dashboard

These instructions walk you through creating an AWS Cloud9 environment, uploading the Product Launch Command Center web files, and previewing the dashboard in your browser.

---

## What You Will Need

- Access to the AWS Management Console (through AWS Academy or your course login)
- The three web files downloaded to your computer:
  - `index.html`
  - `script.js`
  - `style.css`

---

## Part 1: Create a Cloud9 Environment

1. Sign in to the **AWS Management Console**.

2. In the search bar at the top, type **Cloud9** and select **Cloud9** from the results.

3. Click the **Create environment** button.

4. Configure the environment with these settings:

   | Setting | Value |
   |---------|-------|
   | **Name** | `NLP-Web-Dashboard` (or any name you prefer) |
   | **Description** | Product Launch Command Center for NLP analysis |
   | **Environment type** | New EC2 instance |
   | **Instance type** | t2.micro (1 GiB RAM + 1 vCPU) - this is free tier eligible |
   | **Platform** | Amazon Linux 2023 |
   | **Timeout** | 30 minutes |

5. Under **Network settings**, expand the section and set the **Connection** type to **SSH**. This allows Cloud9 to connect directly to your EC2 instance using SSH.

   > **Why SSH?** The SSH connection type opens port 22 on your instance, which is the standard way to securely connect to a remote server. For our purposes it also enables the built-in web preview feature.

6. Leave all other settings at their defaults.

7. Click **Create** at the bottom of the page.

8. Wait for the environment to finish creating. This usually takes 1 to 2 minutes. You will see a green "Successfully created" banner when it is ready.

9. Click **Open** next to your new environment to launch the Cloud9 IDE.

---

## Part 2: Create a Folder for Your Web Files

Once the Cloud9 IDE opens, you will see a file tree on the left and a terminal panel at the bottom.

1. In the file tree on the left, **right-click** on the `environment` folder.

2. Select **New Folder** from the menu that appears.

3. Type `nlp-dashboard` as the folder name and press **Enter**.

4. You should now see a folder called `nlp-dashboard` listed under the `environment` folder in the file tree.

---

## Part 3: Upload the Web Files

1. In the file tree on the left, **click on the `nlp-dashboard` folder** to select it (it should become highlighted).

2. From the menu bar at the top, click **File**, then click **Upload Local Files...**.

3. In the upload dialog that appears, click **Select files**.

4. Navigate to where you saved the three web files on your computer. Select all three files at once:
   - `index.html`
   - `script.js`
   - `style.css`

   > **Tip:** You can select multiple files by holding **Ctrl** (Windows) or **Cmd** (Mac) while clicking each file.

5. Click **Open** to start the upload.

6. Wait for all three files to finish uploading. You will see a confirmation for each file.

7. Close the upload dialog.

8. In the file tree, expand the `nlp-dashboard` folder. You should see all three files listed:

   ```
   environment/
     nlp-dashboard/
       index.html
       script.js
       style.css
   ```

---

## Part 4: Verify Your Files

Let us do a quick check to make sure everything uploaded correctly.

1. In the file tree, **double-click on `index.html`** to open it in the editor. You should see HTML code with a title that says "Product Launch Command Center".

2. Double-click on `script.js` to open it. You should see JavaScript code starting with a comment block about the Product Launch Command Center.

3. Double-click on `style.css` to open it. You should see CSS styling code with custom properties at the top.

If any file looks empty or did not upload correctly, repeat the upload step for that file.

---

## Part 5: Preview the Dashboard

Cloud9 has a built-in web preview feature that lets you see your HTML page directly in the IDE.

### Option A: Preview Inside Cloud9 (Quick Preview)

1. In the file tree, **click once on `index.html`** to select it.

2. From the menu bar, click **Preview**, then click **Preview File**.

   Alternatively, you can click **Preview** and then **Preview Running Application** if the option above is not available.

3. A preview tab will open inside Cloud9 showing your dashboard. You will see the "Product Launch Command Center" header, the business scenario card, and the review input area.

   > **Note:** The preview panel can be small. To make it larger, drag the panel divider or click the **Pop Out Into New Window** button (the square with an arrow icon in the top-right corner of the preview tab). This opens the preview in a full browser tab.

### Option B: Preview in a New Browser Tab (Recommended)

1. In the Cloud9 terminal, run this command to find the preview URL:

   ```bash
   echo "Preview URL: https://$C9_PID.vfs.cloud9.$AWS_REGION.amazonaws.com/nlp-dashboard/index.html"
   ```

2. The terminal will print a URL. **Copy the entire URL**.

3. Open a new browser tab and **paste the URL** into the address bar. Press **Enter**.

4. You should see the full Product Launch Command Center dashboard in your browser.

   > **Important:** This preview URL only works while your Cloud9 environment is running and only works for you (it is tied to your AWS session). It is not a public website.

### What You Should See

When the preview loads successfully, you will see:

- A blue header with "Product Launch Command Center" and a rocket icon
- A "Business Scenario" card explaining the dashboard
- A "Customer Feedback" text area pre-loaded with 8 sample reviews
- An "Analyze Feedback" button and a "Clear Results" button
- An "API Settings" section at the bottom (you will configure this with your endpoint URLs later)

---

## Part 6: Configure Your API Endpoints

The dashboard will not analyze reviews until you connect it to your SageMaker endpoints through API Gateway.

1. Scroll to the bottom of the dashboard page to the **API Settings** section.

2. Paste your API Gateway URLs into the appropriate fields:

   | Field | What to Paste |
   |-------|---------------|
   | **BlazingText API Gateway URL** | Your BlazingText sentiment classification endpoint URL |
   | **LDA POS API Gateway URL** | Your LDA positive topic modeling endpoint URL (optional) |
   | **LDA NEG API Gateway URL** | Your LDA negative topic modeling endpoint URL (optional) |

3. Click **Save Settings**. The button will briefly turn green and say "Saved!" to confirm.

   > Your settings are saved in the browser and will persist across page reloads, so you only need to do this once per browser session.

4. Now click **Analyze Feedback** to test the dashboard with the sample reviews.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Cloud9 environment will not create | Make sure you selected **t2.micro** as the instance type. Larger instances may not be available in your account. |
| Files do not appear after upload | Right-click the `nlp-dashboard` folder in the file tree and select **Refresh**. |
| Preview shows a blank page | Make sure you are previewing `index.html` (not `script.js` or `style.css`). Also check that all three files are in the same folder. |
| Preview URL does not load | Your Cloud9 environment may have timed out. Go back to the Cloud9 console and re-open your environment. |
| "Analyze Feedback" returns an error | Check that your API Gateway URLs are correct in the Settings section and that your SageMaker endpoints are still running. |
| Pop-up blocker warning | Allow pop-ups from the Cloud9 domain if you are using the "Pop Out Into New Window" button. |

---

## Cleaning Up

When you are finished working, your Cloud9 environment will **automatically stop** after the timeout period you set (30 minutes of inactivity). You do not need to manually shut it down.

If you want to delete the environment entirely (to avoid any charges):

1. Go back to the **Cloud9 console** (search for Cloud9 in the AWS search bar).
2. Select your `NLP-Web-Dashboard` environment.
3. Click **Delete**.
4. Type the confirmation phrase and click **Delete** to confirm.

> **Reminder:** Make sure your SageMaker endpoints are also deleted when you are done to avoid ongoing charges. The dashboard's "Analyze Feedback" button will not work after endpoints are deleted, which is expected.
