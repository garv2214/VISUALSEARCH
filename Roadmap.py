import csv

roadmap = [
    ("Week 1", "Understand how blockchains work (hashing, blocks, mining)"),
    ("Week 1", "Install Node.js, VS Code, Git, Metamask"),
    ("Week 1", "Claim ETH on Goerli/Sepolia testnet"),
    ("Week 1", "Try Blockchain Demo (https://andersbrownworth.com/blockchain)"),
    ("Week 2", "Start CryptoZombies course"),
    ("Week 2", "Learn Solidity syntax: variables, mappings, events"),
    ("Week 2", "Deploy a basic contract on Remix"),
    ("Week 3", "Set up Hardhat project"),
    ("Week 3", "Deploy voting contract to localhost"),
    ("Week 3", "Write basic test cases for smart contract"),
    ("Week 4", "Build a simple React app"),
    ("Week 4", "Add Metamask connect button"),
    ("Week 4", "Test Web3.js or Ethers.js calls"),
    ("Week 5", "Create full voting smart contract"),
    ("Week 5", "Add vote counter logic and winner announcement"),
    ("Week 6", "Display candidates from smart contract"),
    ("Week 6", "Allow vote submission via frontend"),
    ("Week 6", "Show frontend feedback messages"),
    ("Week 7", "Listen for smart contract events"),
    ("Week 7", "Display live vote counts"),
    ("Week 7", "Prevent multiple voting per address"),
    ("Week 8", "Set up backend with Express/Flask"),
    ("Week 8", "Connect backend to MongoDB/Firebase"),
    ("Week 8", "Track voting stats or serve admin info"),
    ("Week 9", "Study smart contract vulnerabilities"),
    ("Week 9", "Add access control (e.g., onlyOwner)"),
    ("Week 9", "Optimize gas usage"),
    ("Week 10", "Add voting timer (start/end time)"),
    ("Week 10", "Create admin setup interface"),
    ("Week 10", "Final polish on design and UX"),
    ("Week 11", "Deploy to Goerli/Sepolia using Hardhat"),
    ("Week 11", "Update frontend to point to testnet"),
    ("Week 11", "Host frontend on Netlify/Vercel"),
    ("Week 12", "Write README with setup guide, contract address"),
    ("Week 12", "Record a 2–5 min video demo"),
    ("Week 12", "Optional: Create a simple landing page"),
]

with open("blockchain_voting_roadmap.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Week", "Task"])
    writer.writerows(roadmap)

print("✅ CSV file 'blockchain_voting_roadmap.csv' has been created.")
