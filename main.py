import tkinter as tk
from tkinter import ttk, messagebox
import heapq
import random
import time

#  CONSTANTS
CELL_SIZE   = 30
COLORS = {
    "empty":    "#FFFFFF",
    "wall":     "#2C3E50",
    "start":    "#27AE60",
    "goal":     "#E74C3C",
    "frontier": "#F1C40F",   # yellow
    "visited":  "#3498DB",   # blue
    "path":     "#2ECC71",   # green
}

#  HEURISTICS
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

def chebyshev(a, b):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

HEURISTICS = {
    "Manhattan":  manhattan,
    "Euclidean":  euclidean,
    "Chebyshev":  chebyshev,
}

#  SEARCH ALGORITHMS
def get_neighbors(grid, rows, cols, node):
    r, c = node
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    result = []
    for dr, dc in directions:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
            result.append((nr, nc))
    return result

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    path.reverse()
    return path

def gbfs(grid, rows, cols, start, goal, heuristic_fn):
    """Greedy Best-First Search"""
    h = heuristic_fn
    frontier = []
    heapq.heappush(frontier, (h(start, goal), start))
    came_from = {}
    visited = set()
    visited.add(start)
    frontier_set = {start}
    order = []   # (node, type)  type = 'frontier'|'visited'

    while frontier:
        _, current = heapq.heappop(frontier)
        frontier_set.discard(current)
        order.append((current, 'visited'))

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, order, len(visited)

        for nb in get_neighbors(grid, rows, cols, current):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = current
                heapq.heappush(frontier, (h(nb, goal), nb))
                frontier_set.add(nb)
                order.append((nb, 'frontier'))

    return None, order, len(visited)

def astar(grid, rows, cols, start, goal, heuristic_fn):
    """A* Search"""
    h = heuristic_fn
    frontier = []
    heapq.heappush(frontier, (h(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    closed = set()
    order = []

    while frontier:
        f, g, current = heapq.heappop(frontier)
        if current in closed:
            continue
        closed.add(current)
        order.append((current, 'visited'))

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, order, len(closed)

        for nb in get_neighbors(grid, rows, cols, current):
            tentative_g = g_score[current] + 1
            if nb not in g_score or tentative_g < g_score[nb]:
                g_score[nb] = tentative_g
                f_val = tentative_g + h(nb, goal)
                came_from[nb] = current
                heapq.heappush(frontier, (f_val, tentative_g, nb))
                order.append((nb, 'frontier'))

    return None, order, len(closed)

#  MAIN APP
class PathfindingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Pathfinding Agent - AI 2002")
        self.root.configure(bg="#1A1A2E")

        self.rows = 15
        self.cols = 20
        self.grid = [[0]*self.cols for _ in range(self.rows)]
        self.start = (0, 0)
        self.goal  = (self.rows-1, self.cols-1)
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]]   = 0

        self.mode = "wall"      # wall | start | goal
        self.algorithm  = tk.StringVar(value="A*")
        self.heuristic  = tk.StringVar(value="Manhattan")
        self.density    = tk.DoubleVar(value=0.30)
        self.dynamic_mode = tk.BooleanVar(value=False)
        self.speed      = tk.IntVar(value=30)

        self.nodes_visited = tk.IntVar(value=0)
        self.path_cost     = tk.IntVar(value=0)
        self.exec_time     = tk.StringVar(value="0 ms")

        self.animation_ids = []
        self.current_path  = []
        self.agent_pos     = None
        self.running       = False

        self._build_ui()
        self._draw_grid()

    # ── UI CONSTRUCTION ──────────────────────
    def _build_ui(self):
        # Left panel
        ctrl = tk.Frame(self.root, bg="#16213E", padx=10, pady=10, width=220)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="⚙ Controls", bg="#16213E", fg="#E94560",
                 font=("Arial",13,"bold")).pack(pady=(0,10))

        # Grid size
        self._section(ctrl, "Grid Size")
        row_frame = tk.Frame(ctrl, bg="#16213E"); row_frame.pack(fill=tk.X)
        tk.Label(row_frame, text="Rows:", bg="#16213E", fg="white", width=6).pack(side=tk.LEFT)
        self.rows_spin = tk.Spinbox(row_frame, from_=5, to=30, width=5, textvariable=tk.IntVar(value=15))
        self.rows_spin.pack(side=tk.LEFT)
        col_frame = tk.Frame(ctrl, bg="#16213E"); col_frame.pack(fill=tk.X, pady=2)
        tk.Label(col_frame, text="Cols:", bg="#16213E", fg="white", width=6).pack(side=tk.LEFT)
        self.cols_spin = tk.Spinbox(col_frame, from_=5, to=40, width=5, textvariable=tk.IntVar(value=20))
        self.cols_spin.pack(side=tk.LEFT)
        self._btn(ctrl, "Apply Grid Size", self._apply_grid_size, "#8E44AD")

        # Algorithm
        self._section(ctrl, "Algorithm")
        for alg in ["GBFS", "A*"]:
            tk.Radiobutton(ctrl, text=alg, variable=self.algorithm, value=alg,
                           bg="#16213E", fg="white", selectcolor="#E94560",
                           activebackground="#16213E").pack(anchor=tk.W)

        # Heuristic
        self._section(ctrl, "Heuristic")
        for h in HEURISTICS:
            tk.Radiobutton(ctrl, text=h, variable=self.heuristic, value=h,
                           bg="#16213E", fg="white", selectcolor="#E94560",
                           activebackground="#16213E").pack(anchor=tk.W)

        # Edit mode
        self._section(ctrl, "Edit Mode")
        for m, lbl in [("wall","Draw Walls"),("start","Set Start"),("goal","Set Goal"),("erase","Erase")]:
            tk.Radiobutton(ctrl, text=lbl, variable=tk.StringVar(value="wall"),
                           value=m, bg="#16213E", fg="white", selectcolor="#E94560",
                           activebackground="#16213E",
                           command=lambda v=m: setattr(self, 'mode', v)).pack(anchor=tk.W)

        # Density
        self._section(ctrl, f"Obstacle Density: {int(self.density.get()*100)}%")
        self.density_label = ctrl.winfo_children()[-1]
        s = tk.Scale(ctrl, from_=0, to=60, orient=tk.HORIZONTAL,
                     variable=tk.IntVar(value=30), bg="#16213E", fg="white",
                     highlightthickness=0, troughcolor="#E94560",
                     command=lambda v: [self.density.set(int(v)/100),
                                        self.density_label.config(text=f"Obstacle Density: {v}%")])
        s.pack(fill=tk.X)

        # Speed
        self._section(ctrl, "Animation Speed")
        tk.Scale(ctrl, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.speed,
                 bg="#16213E", fg="white", highlightthickness=0,
                 troughcolor="#27AE60").pack(fill=tk.X)

        # Dynamic mode
        self._section(ctrl, "Dynamic Obstacles")
        tk.Checkbutton(ctrl, text="Enable Dynamic Mode", variable=self.dynamic_mode,
                       bg="#16213E", fg="white", selectcolor="#E94560",
                       activebackground="#16213E").pack(anchor=tk.W)

        # Action buttons
        self._section(ctrl, "Actions")
        self._btn(ctrl, "▶ Start Search",   self._start_search,  "#E94560")
        self._btn(ctrl, "⟳ Random Map",     self._random_map,    "#2980B9")
        self._btn(ctrl, "✕ Clear Grid",     self._clear_grid,    "#7F8C8D")

        # Metrics
        self._section(ctrl, "Metrics")
        for lbl, var in [("Nodes Visited:", self.nodes_visited),
                         ("Path Cost:",     self.path_cost)]:
            f = tk.Frame(ctrl, bg="#16213E"); f.pack(fill=tk.X)
            tk.Label(f, text=lbl, bg="#16213E", fg="#BDC3C7", width=13, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(f, textvariable=var, bg="#16213E", fg="#F1C40F", font=("Arial",10,"bold")).pack(side=tk.LEFT)
        f = tk.Frame(ctrl, bg="#16213E"); f.pack(fill=tk.X)
        tk.Label(f, text="Exec Time:", bg="#16213E", fg="#BDC3C7", width=13, anchor=tk.W).pack(side=tk.LEFT)
        tk.Label(f, textvariable=self.exec_time, bg="#16213E", fg="#F1C40F", font=("Arial",10,"bold")).pack(side=tk.LEFT)

        # Legend
        self._section(ctrl, "Legend")
        for color, label in COLORS.items():
            if color in ("empty","wall","start","goal","frontier","visited","path"):
                f = tk.Frame(ctrl, bg="#16213E"); f.pack(fill=tk.X, pady=1)
                tk.Label(f, bg=COLORS[color], width=3).pack(side=tk.LEFT, padx=(0,5))
                tk.Label(f, text=color.capitalize(), bg="#16213E", fg="white").pack(side=tk.LEFT)

        # Canvas
        canvas_frame = tk.Frame(self.root, bg="#1A1A2E")
        canvas_frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(canvas_frame, text="Dynamic Pathfinding Agent",
                 bg="#1A1A2E", fg="#E94560", font=("Arial",14,"bold")).pack()
        self.canvas = tk.Canvas(canvas_frame,
                                width=self.cols*CELL_SIZE,
                                height=self.rows*CELL_SIZE,
                                bg="#FFFFFF", highlightthickness=2,
                                highlightbackground="#E94560")
        self.canvas.pack()
        self.canvas.bind("<Button-1>",        self._on_click)
        self.canvas.bind("<B1-Motion>",       self._on_drag)

    def _section(self, parent, text):
        tk.Label(parent, text=text, bg="#16213E", fg="#ECF0F1",
                 font=("Arial",9,"bold")).pack(anchor=tk.W, pady=(8,2))

    def _btn(self, parent, text, cmd, color="#E94560"):
        tk.Button(parent, text=text, command=cmd, bg=color, fg="white",
                  relief=tk.FLAT, font=("Arial",9,"bold"),
                  activebackground=color, cursor="hand2").pack(fill=tk.X, pady=2)

    # ── GRID ────────────────────────────────
    def _apply_grid_size(self):
        self._stop()
        self.rows = int(self.rows_spin.get())
        self.cols = int(self.cols_spin.get())
        self.grid = [[0]*self.cols for _ in range(self.rows)]
        self.start = (0, 0)
        self.goal  = (self.rows-1, self.cols-1)
        self.canvas.config(width=self.cols*CELL_SIZE, height=self.rows*CELL_SIZE)
        self._draw_grid()

    def _draw_grid(self):
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                self._draw_cell(r, c)

    def _draw_cell(self, r, c, color=None):
        x1, y1 = c*CELL_SIZE, r*CELL_SIZE
        x2, y2 = x1+CELL_SIZE, y1+CELL_SIZE
        if color is None:
            if (r,c) == self.start:     color = COLORS["start"]
            elif (r,c) == self.goal:    color = COLORS["goal"]
            elif self.grid[r][c] == 1:  color = COLORS["wall"]
            else:                       color = COLORS["empty"]
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color,
                                     outline="#CCCCCC", width=1)
        if (r,c) == self.start:
            self.canvas.create_text(x1+CELL_SIZE//2, y1+CELL_SIZE//2,
                                    text="S", fill="white", font=("Arial",9,"bold"))
        elif (r,c) == self.goal:
            self.canvas.create_text(x1+CELL_SIZE//2, y1+CELL_SIZE//2,
                                    text="G", fill="white", font=("Arial",9,"bold"))

    def _cell_from_event(self, event):
        c = event.x // CELL_SIZE
        r = event.y // CELL_SIZE
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        return None

    def _on_click(self, event):
        cell = self._cell_from_event(event)
        if not cell: return
        r, c = cell
        if self.mode == "start":
            self.start = (r, c); self.grid[r][c] = 0
        elif self.mode == "goal":
            self.goal  = (r, c); self.grid[r][c] = 0
        elif self.mode == "erase":
            self.grid[r][c] = 0
        else:
            if (r,c) not in (self.start, self.goal):
                self.grid[r][c] = 1
        self._draw_grid()

    def _on_drag(self, event):
        cell = self._cell_from_event(event)
        if not cell: return
        r, c = cell
        if self.mode == "wall" and (r,c) not in (self.start, self.goal):
            self.grid[r][c] = 1
            self._draw_cell(r, c)
        elif self.mode == "erase":
            self.grid[r][c] = 0
            self._draw_cell(r, c)

    def _random_map(self):
        self._stop()
        self.grid = [[0]*self.cols for _ in range(self.rows)]
        density = self.density.get()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c) not in (self.start, self.goal):
                    self.grid[r][c] = 1 if random.random() < density else 0
        self._draw_grid()

    def _clear_grid(self):
        self._stop()
        self.grid = [[0]*self.cols for _ in range(self.rows)]
        self.nodes_visited.set(0)
        self.path_cost.set(0)
        self.exec_time.set("0 ms")
        self._draw_grid()

    # ── SEARCH ──────────────────────────────
    def _start_search(self):
        self._stop()
        self._draw_grid()
        alg = self.algorithm.get()
        h_fn = HEURISTICS[self.heuristic.get()]

        t0 = time.time()
        if alg == "GBFS":
            path, order, n_visited = gbfs(self.grid, self.rows, self.cols,
                                          self.start, self.goal, h_fn)
        else:
            path, order, n_visited = astar(self.grid, self.rows, self.cols,
                                           self.start, self.goal, h_fn)
        elapsed = (time.time() - t0) * 1000

        self.nodes_visited.set(n_visited)
        self.exec_time.set(f"{elapsed:.2f} ms")

        if path is None:
            messagebox.showinfo("No Path", "No path found to the goal!")
            return

        self.path_cost.set(len(path) - 1)
        self.current_path = path
        self.running = True
        self._animate(order, path, 0)

    def _animate(self, order, path, idx):
        if not self.running:
            return
        delay = max(1, 101 - self.speed.get())
        if idx < len(order):
            node, ntype = order[idx]
            if node not in (self.start, self.goal):
                color = COLORS["frontier"] if ntype == "frontier" else COLORS["visited"]
                self._draw_cell(*node, color=color)
            aid = self.root.after(delay, self._animate, order, path, idx+1)
            self.animation_ids.append(aid)
        else:
            # Draw final path
            for node in path:
                if node not in (self.start, self.goal):
                    self._draw_cell(*node, color=COLORS["path"])
            self._draw_cell(*self.start)
            self._draw_cell(*self.goal)
            # Dynamic mode
            if self.dynamic_mode.get():
                self._start_dynamic(path)

    def _start_dynamic(self, path):
        """Spawn obstacles; re-plan if path is blocked."""
        def step(pos_idx):
            if not self.running or pos_idx >= len(path):
                return
            pos = path[pos_idx]
            # Spawn random obstacle
            if random.random() < 0.15:
                candidates = [(r,c) for r in range(self.rows)
                              for c in range(self.cols)
                              if self.grid[r][c]==0 and (r,c) not in (self.start, self.goal, pos)]
                if candidates:
                    obs = random.choice(candidates)
                    self.grid[obs[0]][obs[1]] = 1
                    self._draw_cell(*obs, color=COLORS["wall"])
                    # Check if obstacle is on current path
                    remaining = path[pos_idx:]
                    if obs in remaining:
                        # Re-plan from current position
                        h_fn = HEURISTICS[self.heuristic.get()]
                        alg  = self.algorithm.get()
                        if alg == "GBFS":
                            new_path, new_order, nv = gbfs(self.grid, self.rows, self.cols,
                                                           pos, self.goal, h_fn)
                        else:
                            new_path, new_order, nv = astar(self.grid, self.rows, self.cols,
                                                            pos, self.goal, h_fn)
                        self.nodes_visited.set(self.nodes_visited.get() + nv)
                        if new_path:
                            self.path_cost.set(len(new_path)-1)
                            for n in new_path:
                                if n not in (self.start, self.goal):
                                    self._draw_cell(*n, color=COLORS["path"])
                            aid = self.root.after(50, step, 0)
                            self.animation_ids.append(aid)
                        else:
                            messagebox.showinfo("Blocked", "No path available after obstacle spawn!")
                        return
            # Move agent
            if pos not in (self.start, self.goal):
                self._draw_cell(*pos, color=COLORS["visited"])
            aid = self.root.after(200, step, pos_idx+1)
            self.animation_ids.append(aid)

        step(0)

    def _stop(self):
        self.running = False
        for aid in self.animation_ids:
            self.root.after_cancel(aid)
        self.animation_ids.clear()

if __name__ == "__main__":
    root = tk.Tk()
    app  = PathfindingApp(root)
    root.mainloop()
