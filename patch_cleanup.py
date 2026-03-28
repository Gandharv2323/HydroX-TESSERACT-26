import re

with open('dashboard.html', 'rb') as f:
    text = f.read().decode('utf-8', errors='ignore')

# 1. Clean up garbled emojis in Scenario Control
text = re.sub(r'<span[^>]*?>.*?</span>(?=\s*<br>NORMAL)', '<span style="font-size:16px;">🟢</span>', text)
text = re.sub(r'<span[^>]*?>.*?</span>(?=\s*<br>CAVITATION)', '<span style="font-size:16px;">💥</span>', text)
text = re.sub(r'<span[^>]*?>.*?</span>(?=\s*<br>BEARING WEAR)', '<span style="font-size:16px;">⚙️</span>', text)
text = re.sub(r'<span[^>]*?>.*?</span>(?=\s*<br>DRY RUN)', '<span style="font-size:16px;">🏜️</span>', text)

# 2. Clean up garbled text in alert banner JS (where there are switch cases)
text = text.replace('fm==="cavitation"?"💥 CAVITATION"', 'fm==="cavitation"?"💥 CAVITATION"')
# We just forcefully fix the alert banner JS text assignment
# It might be: fm==="cavitation"?"Â§Â CAVITATION"...
text = re.sub(r'bannerMode".textContent = .*?;', 
    'bannerMode".textContent = fm==="cavitation"?"💥 CAVITATION":fm==="bearing_wear"?"⚙️ BEARING WEAR":fm==="dry_run"?"🏜️ DRY RUN":"FAULT DETECTED";', text)

# 3. Clean up the `â€”` (em dash) in event logs
text = text.replace('Anomaly cleared â€” normal', 'Anomaly cleared — normal')
text = text.replace('Health CRITICAL â€” action required', 'Health CRITICAL — action required')
text = text.replace('Mode â†’ ', 'Mode → ')
text = text.replace('â€"', '—')

# Also fix the initial alertAction text
text = re.sub(r'id=\"alertAction\">.*?</span>', 'id="alertAction">—</span>', text)

# 4. Make sure alert banner doesn't have 'visible' by default in HTML
text = text.replace('class="alert-banner visible"', 'class="alert-banner"')

# 5. Fix UI text that was ruined
text = text.replace('Smart Pump Digital Twin â€” Trimiti Innovations P-09', 'Smart Pump Digital Twin — Trimiti Innovations P-09')
text = text.replace('computing&hellip;', 'computing...')

# 6. Fix any residual JS syntax errors (d_maint handles the variable shadowing we fixed earlier)
if 'var d_maint =' not in text and 'var d = new Date' in text:
    text = text.replace('var d = new Date(Date', 'var d_maint = new Date(Date')
    text = text.replace('maintDate.textContent = d.to', 'maintDate.textContent = d_maint.to')

# 7. Check if there are other syntax errors like missing closing brackets from the previous patches
# The node script said ReferenceError: document is not defined meaning no syntax errors were present! So we are good.

with open('dashboard.html', 'wb') as f:
    f.write(text.encode('utf-8'))

print("DASHBOARD CLEANUP COMPLETE!")
