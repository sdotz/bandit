# Implementations of bandit algorithms in Go

## Usage
Pretend we have a site selling tools. On the homepage we feature an ad for one of our tool products, with a button to buy it.

```go

//first define the arms of your bandit. This is the list of tools that could be featured.
var tools := map[int]string{
    0: "shovel",
    1: "axe",
    2: "screwdriver",
    3: "wrench",
    4: "hammer",
    5: "saw",
}

//create the bandit, and initialize it with the number of arms
bandit := ucb.NewUCB1(len(tools))

//"pull" an arm to ask the bandit which tool to display. It returns an int that should correspond to one of your choices. 
pulled := bandit.Pull()

//if the user buys that tool, reward the arm
bandit.Reward(pulled)

//Done! A simple bandit algorithm will learn which tools are selling best, and make sure more people are presented that tool in the future (the pull).
```