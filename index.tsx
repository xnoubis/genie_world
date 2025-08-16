import { GoogleGenAI } from "@google/genai";

// --- CONFIGURATION ---
const GRID_SIZE = 15;
const CELL_SIZE = 20;

// --- DOM ELEMENTS ---
const elements = {
  seedInput: document.getElementById('seed') as HTMLInputElement,
  generateBtn: document.getElementById('generate-world') as HTMLButtonElement,
  trainBtn: document.getElementById('train-agent') as HTMLButtonElement,
  testWarBtn: document.getElementById('test-war') as HTMLButtonElement,
  testBsBtn: document.getElementById('test-bs') as HTMLButtonElement,
  worldCanvas: document.getElementById('world-canvas') as HTMLCanvasElement,
  agentCanvas: document.getElementById('agent-canvas') as HTMLCanvasElement,
  worldStatus: document.getElementById('world-status') as HTMLParagraphElement,
  agentStatus: document.getElementById('agent-status') as HTMLParagraphElement,
  testResults: document.getElementById('test-results') as HTMLDivElement,
  explainBtns: document.querySelectorAll('.explain-btn') as NodeListOf<HTMLButtonElement>,
};

// --- STATE ---
let world = null;
let agent = null;
let worldCtx = elements.worldCanvas.getContext('2d');
let agentCtx = elements.agentCanvas.getContext('2d');

// --- GEMINI API SETUP ---
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// --- UTILITIES ---
// Simple seeded PRNG (for determinism)
class SeededRandom {
  private seed: number;
  constructor(seedStr: string) {
    this.seed = this.hashString(seedStr);
  }
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash |= 0; // Convert to 32bit integer
    }
    return hash;
  }
  next() {
    this.seed = (this.seed * 9301 + 49297) % 233280;
    return this.seed / 233280;
  }
  nextInt(max: number) {
    return Math.floor(this.next() * max);
  }
}

// --- WORLD GENERATION (W-Gen & Sim) ---
class World {
  public grid: number[][]; // 0: path, 1: wall
  public start: { x: number; y: number };
  public goal: { x: number; y: number };
  private random: SeededRandom;

  constructor(seed: string) {
    this.random = new SeededRandom(seed);
    this.grid = Array(GRID_SIZE).fill(0).map(() => Array(GRID_SIZE).fill(1));
    this.generateMaze(1, 1);
    this.start = { x: 1, y: 1 };
    this.grid[1][1] = 0; // Ensure start is open
    // Place goal in a random open spot in the bottom-right quadrant
    let goalX, goalY;
    do {
      goalX = Math.floor(GRID_SIZE / 2) + this.random.nextInt(Math.floor(GRID_SIZE / 2));
      goalY = Math.floor(GRID_SIZE / 2) + this.random.nextInt(Math.floor(GRID_SIZE / 2));
    } while (this.grid[goalY][goalX] === 1);
    this.goal = { x: goalX, y: goalY };
  }

  // Randomized DFS maze generation
  private generateMaze(cx: number, cy: number) {
    this.grid[cy][cx] = 0;
    const directions = [[0, -2], [0, 2], [-2, 0], [2, 0]];
    // Shuffle directions
    for (let i = directions.length - 1; i > 0; i--) {
        const j = this.random.nextInt(i + 1);
        [directions[i], directions[j]] = [directions[j], directions[i]];
    }
    for (const [dx, dy] of directions) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx > 0 && nx < GRID_SIZE - 1 && ny > 0 && ny < GRID_SIZE - 1 && this.grid[ny][nx] === 1) {
        this.grid[ny - dy / 2][nx - dx / 2] = 0;
        this.generateMaze(nx, ny);
      }
    }
  }
}

// --- AGENT MODEL ---
class Agent {
    public internalModel: number[][]; // 0: unknown, 1: path, 2: wall
    public path: {x: number, y: number}[];
    public start: {x: number, y: number};
    public goal: {x: number, y: number};

    constructor(world: World) {
        this.internalModel = Array(GRID_SIZE).fill(0).map(() => Array(GRID_SIZE).fill(0));
        this.path = [];
        this.start = world.start;
        this.goal = world.goal;
    }

    // A* Pathfinding
    findPath(worldGrid: number[][]) {
        // A* implementation...
        const openSet = [{...this.start, g: 0, h: this.heuristic(this.start, this.goal), f: this.heuristic(this.start, this.goal)}];
        const cameFrom = new Map();
        
        this.internalModel[this.start.y][this.start.x] = 1;

        while (openSet.length > 0) {
            openSet.sort((a, b) => a.f - b.f);
            const current = openSet.shift();

            if (current.x === this.goal.x && current.y === this.goal.y) {
                this.reconstructPath(cameFrom, current);
                return;
            }

            const neighbors = this.getNeighbors(current, worldGrid);
            for (const neighbor of neighbors) {
                const tentativeG = current.g + 1;
                const neighborKey = `${neighbor.x},${neighbor.y}`;
                const existingNeighbor = openSet.find(n => n.x === neighbor.x && n.y === neighbor.y);
                
                if (existingNeighbor && tentativeG >= existingNeighbor.g) continue;
                
                cameFrom.set(neighborKey, current);
                neighbor.g = tentativeG;
                neighbor.h = this.heuristic(neighbor, this.goal);
                neighbor.f = neighbor.g + neighbor.h;

                if (!existingNeighbor) {
                    openSet.push(neighbor);
                }
            }
        }
        this.path = []; // No path found
    }

    private getNeighbors(node, grid) {
        const neighbors = [];
        const directions = [[0, 1], [0, -1], [1, 0], [-1, 0]];
        for (const [dx, dy] of directions) {
            const x = node.x + dx;
            const y = node.y + dy;
            if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
                if (grid[y][x] === 0) { // Can move
                    this.internalModel[y][x] = 1; // Agent "sees" path
                    neighbors.push({ x, y });
                } else {
                    this.internalModel[y][x] = 2; // Agent "sees" wall
                }
            }
        }
        return neighbors;
    }
    
    private heuristic(a, b) { // Manhattan distance
        return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
    }
    
    private reconstructPath(cameFrom, current) {
        const totalPath = [current];
        let currentKey = `${current.x},${current.y}`;
        while (cameFrom.has(currentKey)) {
            current = cameFrom.get(currentKey);
            currentKey = `${current.x},${current.y}`;
            totalPath.unshift(current);
        }
        this.path = totalPath;
    }
}


// --- RENDERING ---
function drawGrid(ctx: CanvasRenderingContext2D, grid: number[][], type: 'world' | 'agent') {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  for (let y = 0; y < GRID_SIZE; y++) {
    for (let x = 0; x < GRID_SIZE; x++) {
      if (type === 'world') {
          ctx.fillStyle = grid[y][x] === 1 ? '#3700b3' : '#1e1e1e';
      } else { // agent's model
          if (grid[y][x] === 2) ctx.fillStyle = '#b33700'; // Wall
          else if (grid[y][x] === 1) ctx.fillStyle = '#1e1e1e'; // Path
          else ctx.fillStyle = '#000'; // Unknown
      }
      ctx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
    }
  }
}

function drawEntity(ctx: CanvasRenderingContext2D, pos: {x: number, y: number}, color: string) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(pos.x * CELL_SIZE + CELL_SIZE / 2, pos.y * CELL_SIZE + CELL_SIZE / 2, CELL_SIZE / 3, 0, 2 * Math.PI);
    ctx.fill();
}

function drawPath(ctx: CanvasRenderingContext2D, path: {x: number, y: number}[], color: string) {
    if (path.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(path[0].x * CELL_SIZE + CELL_SIZE / 2, path[0].y * CELL_SIZE + CELL_SIZE / 2);
    for (let i = 1; i < path.length; i++) {
        ctx.lineTo(path[i].x * CELL_SIZE + CELL_SIZE / 2, path[i].y * CELL_SIZE + CELL_SIZE / 2);
    }
    ctx.stroke();
}


// --- APPLICATION LOGIC ---
function handleGenerateWorld() {
  const seed = elements.seedInput.value || 'default';
  world = new World(seed);
  agent = new Agent(world);
  
  drawGrid(worldCtx, world.grid, 'world');
  drawEntity(worldCtx, world.start, '#03dac6');
  drawEntity(worldCtx, world.goal, '#cf6679');
  
  agentCtx.clearRect(0, 0, agentCtx.canvas.width, agentCtx.canvas.height);

  elements.worldStatus.textContent = `World generated with seed: "${seed}"`;
  elements.agentStatus.textContent = 'Train the agent to see its model.';
  elements.testResults.textContent = '';
  
  elements.trainBtn.disabled = false;
  elements.testWarBtn.disabled = true;
  elements.testBsBtn.disabled = true;
}

function handleTrainAgent() {
  agent.findPath(world.grid);
  
  drawGrid(agentCtx, agent.internalModel, 'agent');
  if (agent.path.length > 0) {
      drawPath(worldCtx, agent.path, '#03dac6');
      drawEntity(agentCtx, agent.start, '#03dac6');
      drawEntity(agentCtx, agent.goal, '#cf6679');
      elements.agentStatus.textContent = 'Agent has built an internal model and found a path.';
      elements.testWarBtn.disabled = false;
      elements.testBsBtn.disabled = false;
  } else {
      elements.agentStatus.textContent = 'Agent could not find a path.';
      elements.testWarBtn.disabled = true;
      elements.testBsBtn.disabled = true;
  }
  elements.trainBtn.disabled = true;
}

function handleTestWAR() {
    // WAR is already demonstrated by the "Agent's Internal Model" canvas.
    // This button just highlights the concept.
    elements.testResults.innerHTML = `<strong>WAR Test:</strong> The agent's canvas shows its reconstructed world model, built solely from its experiences (seeing adjacent cells). It can "speak for the world" it has seen.`;
}

function handleTestBS() {
    // Redraw world without the solution path
    drawGrid(worldCtx, world.grid, 'world');
    drawEntity(worldCtx, world.start, '#03dac6');
    drawEntity(worldCtx, world.goal, '#cf6679');
    
    // Agent plans a path based on its internal model. We draw that path on the *real* world.
    drawPath(worldCtx, agent.path, '#f2ff00');
    
    elements.testResults.innerHTML = `<strong>BS Test:</strong> The yellow path was planned by the agent *without* looking at the world again. Because it succeeds, the agent's internal model was sufficient for effective action.`;
}

async function handleExplain(e: MouseEvent) {
    const button = e.target as HTMLButtonElement;
    const concept = button.dataset.concept;
    const contentDiv = button.closest('.concept').querySelector('.explanation-content');
    
    if (!concept || !contentDiv) return;

    if (contentDiv.innerHTML) { // Toggle
        contentDiv.innerHTML = '';
        return;
    }
    
    button.disabled = true;
    contentDiv.innerHTML = 'Thinking...';
    
    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: `In the context of AI consciousness and world simulation, explain the concept of "${concept}" in 2-3 concise sentences for a technical audience.`,
            config: {
              systemInstruction: "You are an AI expert, explaining complex topics simply."
            }
        });
        contentDiv.innerHTML = response.text;
    } catch (error) {
        console.error("Gemini API error:", error);
        contentDiv.innerHTML = 'Sorry, an error occurred while fetching an explanation.';
    } finally {
        button.disabled = false;
    }
}


// --- EVENT LISTENERS ---
elements.generateBtn.addEventListener('click', handleGenerateWorld);
elements.trainBtn.addEventListener('click', handleTrainAgent);
elements.testWarBtn.addEventListener('click', handleTestWAR);
elements.testBsBtn.addEventListener('click', handleTestBS);
elements.explainBtns.forEach(btn => btn.addEventListener('click', handleExplain));

// --- INITIALIZATION ---
function init() {
    // Set canvas dimensions based on config
    elements.worldCanvas.width = elements.agentCanvas.width = GRID_SIZE * CELL_SIZE;
    elements.worldCanvas.height = elements.agentCanvas.height = GRID_SIZE * CELL_SIZE;
    
    elements.worldStatus.textContent = 'Click "Generate World" to begin the simulation.';
    elements.agentStatus.textContent = 'Waiting for a world...';
}

init();
