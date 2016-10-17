# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Things to consider:
        #   - Scared Timer (the higher the better)
        #   - New Ghost States (the further from the ghost the better - Mann. Dist.)
        #   - New Food (the closer to the food the better)
        #   - Remember to avoid walls!!!

        heuristic = 0 
        for states in newScaredTimes:
          heuristic+=states
        for ghost in newGhostStates:
          ghostDist += [manhattanDistance(ghost.getPosition(), newPos)]
          if ghost.getDirection() == Directions.STOP:
            heuristic -= 20
        minGhost = min(ghostDist)
        foodList = newFood.asList()
        foodDist = []
        for food in foodList:
          foodDist += [manhattanDistance(food, newPos)]
        if currentGameState.getNumFood() > successorGameState.getNumFood():
          heuristic += 100
        inverse = 0
        if len(foodDist) > 0:
          inverse = (1.0/float(min(foodDist)))
        if minGhost < 5:
          minGhost = minGhost*0.5
        if minGhost >= 5 and minGhost <= 20:
          minGhost = minGhost*1.1
        if minGhost > 20:
          minGhost = minGhost*1.7
        heuristic += minGhost + successorGameState.getScore() + inverse*100
        return heuristic

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        result = self.value(gameState, 0)
        return result[1]

    def value(self, gameState, depth):
        numAgents = gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose() or (depth == self.depth * numAgents):
          return (self.evaluationFunction(gameState), None)

        if ((depth % numAgents) == 0): #next agent is pacman so max
          return self.minOrMax(gameState, depth, 1)
        else: #next agent is ghost so min
          return self.minOrMax(gameState, depth, 0)  

    def minOrMax(self, gameState, depth, pacman):
        agent = [depth%gameState.getNumAgents(), 0]
        values = [float("inf"), float("-inf")]
        val = (values[pacman], None)
        possibleActions = gameState.getLegalActions(agent[pacman])
        if len(possibleActions) == 0:
          return (self.evaluationFunction(gameState), None)
        for action in possibleActions:
          successor = gameState.generateSuccessor(agent[pacman], action)
          result = self.value(successor, depth+1)
          if ((pacman==1) and result[0] > val[0]) or ((pacman==0) and (result[0] < val[0])):
            val = (result[0], action)
        return val


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #alpha starts at negative (maximizing), beta starts at positive (minimizing)
        result = self.value(gameState, 0, -float("inf"), float("inf"))
        return result[1]

    def value(self, gameState, depth, alpha, beta):
        numAgents = gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose() or (depth == self.depth * numAgents):
          return (self.evaluationFunction(gameState), None)

        if ((depth % numAgents) == 0): #next agent is pacman so max
          return self.minOrMax(gameState, depth, 1, alpha, beta)
        else: #next agent is ghost so min
          return self.minOrMax(gameState, depth, 0, alpha, beta)  

    def minOrMax(self, gameState, depth, pacman, alpha, beta):
        agent = [depth%gameState.getNumAgents(), 0]
        values = [float("inf"), float("-inf")]
        val = (values[pacman], None)
        possibleActions = gameState.getLegalActions(agent[pacman])
        if len(possibleActions) == 0:
          return (self.evaluationFunction(gameState), None)
        for action in possibleActions:
          successor = gameState.generateSuccessor(agent[pacman], action)
          result = self.value(successor, depth+1, alpha, beta)
          if pacman:
            if result[0] > val[0]:
              val = (result[0], action)
            if val[0] > beta:
              return val
            alpha = max(alpha, val[0])
          else:
            if result[0] < val[0]:
              val = (result[0], action)
            if val[0] < alpha:
              return val
            beta = min(beta, val[0])
        return val

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        result = self.value(gameState, 0)
        return result[1]

    def value(self, gameState, depth):
        numAgents = gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose() or (depth == self.depth * numAgents):
          return (self.evaluationFunction(gameState), None)

        if ((depth % numAgents) == 0): #next agent is pacman so max
          return self.expectOrMax(gameState, depth, 1)
        else: #next agent is ghost so min
          return self.expectOrMax(gameState, depth, 0)  

    def expectOrMax(self, gameState, depth, pacman):
        agent = [depth%gameState.getNumAgents(), 0]
        values = [0, float("-inf")]
        val = (values[pacman], None)
        possibleActions = gameState.getLegalActions(agent[pacman])
        if len(possibleActions) == 0:
          return (self.evaluationFunction(gameState), None)
        for action in possibleActions:
          successor = gameState.generateSuccessor(agent[pacman], action)
          result = self.value(successor, depth+1)
          if pacman:
            if result[0] > val[0]:
              val = (result[0], action)
          else:
              p = 1.0/float(len(possibleActions))
              val = (val[0]+(result[0]*p), action)
        return val

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    heuristic = 0 
    for states in newScaredTimes:
      heuristic+=states
    ghostDist = []
    for ghost in newGhostStates:
      ghostDist += [manhattanDistance(ghost.getPosition(), newPos)]
      if ghost.getDirection() == Directions.STOP:
        heuristic -= 20
    minGhost = min(ghostDist)
    foodList = newFood.asList()
    foodDist = []
    for food in foodList:
      foodDist += [manhattanDistance(food, newPos)]
    if currentGameState.getNumFood() > successorGameState.getNumFood():
      heuristic += 100
    inverse = 0
    if len(foodDist) > 0:
      inverse = (1.0/float(min(foodDist)))
    if minGhost < 5:
      minGhost = minGhost*0.5
    if minGhost >= 5 and minGhost <= 20:
      minGhost = minGhost*1.1
    if minGhost > 20:
      minGhost = minGhost*1.7
    heuristic += minGhost + successorGameState.getScore() + inverse*100
    return heuristic
    

# Abbreviation
better = betterEvaluationFunction

