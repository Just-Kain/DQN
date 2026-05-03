using System;
using System.Collections.Generic;

// ──────────────────────────────────────────────────────────────────────────────
// BSP Dungeon Generator
// Алгоритм:
//   1. Делим пространство бинарным деревом (BSP) на листья-секции.
//   2. В каждом листе размещаем комнату случайного размера.
//   3. Соединяем соседние комнаты Г-образными коридорами.
//   4. Ставим Exit в центр случайной комнаты.
//   5. Окружаем Exit ямами (Pit) — только по диагоналям,
//      чтобы оставить 4 кардинальных подхода свободными.
//   6. BFS от Exit → находим клетку с максимальным расстоянием
//      (там будет спавн игрока).
// ──────────────────────────────────────────────────────────────────────────────
public class DungeonGenerator
{
    // ── Настройки BSP ──────────────────────────────────────────────────────────
    private const int MinSplitSize = 4;   // минимальный размер секции для разбиения
    private const int MaxDepth     = 4;   // максимальная глубина дерева
    private const int MinRoomSize  = 4;   // минимальный размер комнаты

    private Random rng = null!;

    // ── Публичный API ──────────────────────────────────────────────────────────
    public DungeonMap Generate(int width, int height, int seed)
    {
        rng = new Random(seed);

        var map = new DungeonMap { Tiles = new TileType[width, height] };

        // 1. Заполнить всё стенами
        FillAll(map, TileType.Wall);

        // 2. Построить BSP-дерево (interior = [1..width-2] x [1..height-2])
        var root = new BspNode(1, 1, width - 2, height - 2);
        Split(root, 0);

        // 3. Вырезать комнаты в листьях
        CarveRooms(map, root);

        // 4. Соединить комнаты коридорами
        CarveCorridors(map, root);

        // 5. Собрать комнаты и сохранить в карту
        var leaves = CollectLeaves(root);
        foreach (var leaf in leaves)
            map.Rooms.Add((leaf.RoomX, leaf.RoomY, leaf.RoomW, leaf.RoomH));

        // Разместить Exit в случайной комнате
        var exitLeaf = leaves[rng.Next(leaves.Count)];
        int exitX = exitLeaf.RoomX + exitLeaf.RoomW / 2;
        int exitY = exitLeaf.RoomY + exitLeaf.RoomH / 2;
        map.Tiles[exitX, exitY] = TileType.Exit;
        map.ExitX = exitX;
        map.ExitY = exitY;

        // 6. Ямы в комнатах: у стен и в центрах
        PlacePitsInRooms(map, leaves, exitX, exitY);

        // 7. BFS → позиция спавна игрока
        var (spawnX, spawnY) = FindMaxDistanceSpawn(map, exitX, exitY);
        
        map.Tiles[spawnX, spawnY] = TileType.Floor;

        map.PlayerSpawnX = spawnX;
        map.PlayerSpawnY = spawnY;

        return map;
    }

    // ── Внутренние методы ──────────────────────────────────────────────────────

    private static void FillAll(DungeonMap map, TileType tile)
    {
        for (int x = 0; x < map.Width; x++)
            for (int y = 0; y < map.Height; y++)
                map.Tiles[x, y] = tile;
    }

    // ── BSP: рекурсивное разбиение ─────────────────────────────────────────────
    private void Split(BspNode node, int depth)
    {
        if (depth >= MaxDepth) return;

        bool canSplitH = node.H >= MinSplitSize * 2;
        bool canSplitV = node.W >= MinSplitSize * 2;
        if (!canSplitH && !canSplitV) return;

        // Выбираем направление разбиения
        bool splitHorizontal;
        if (canSplitH && canSplitV)
            splitHorizontal = rng.Next(2) == 0;
        else
            splitHorizontal = canSplitH;

        if (splitHorizontal)
        {
            // Горизонтальный разрез: фиксируем y-координату разреза
            int min = node.Y + MinSplitSize;
            int max = node.Y + node.H - MinSplitSize;
            if (min >= max) return;
            int cut = rng.Next(min, max);

            node.Left  = new BspNode(node.X, node.Y, node.W, cut - node.Y);
            node.Right = new BspNode(node.X, cut,    node.W, node.Y + node.H - cut);
        }
        else
        {
            // Вертикальный разрез: фиксируем x-координату разреза
            int min = node.X + MinSplitSize;
            int max = node.X + node.W - MinSplitSize;
            if (min >= max) return;
            int cut = rng.Next(min, max);

            node.Left  = new BspNode(node.X, node.Y, cut - node.X,           node.H);
            node.Right = new BspNode(cut,    node.Y, node.X + node.W - cut,  node.H);
        }

        Split(node.Left!,  depth + 1);
        Split(node.Right!, depth + 1);
    }

    // ── Вырезка комнат в листьях ───────────────────────────────────────────────
    private void CarveRooms(DungeonMap map, BspNode node)
    {
        if (node.IsLeaf)
        {
            // Размеры комнаты: от MinRoomSize до (размер секции - 2)
            int maxW = Math.Max(MinRoomSize, node.W - 2);
            int maxH = Math.Max(MinRoomSize, node.H - 2);

            int rw = rng.Next(MinRoomSize, maxW + 1);
            int rh = rng.Next(MinRoomSize, maxH + 1);

            // Убеждаемся, что комната не выходит за пределы секции
            rw = Math.Min(rw, node.W);
            rh = Math.Min(rh, node.H);

            int maxOffX = node.W - rw;
            int maxOffY = node.H - rh;

            int rx = node.X + (maxOffX > 0 ? rng.Next(0, maxOffX + 1) : 0);
            int ry = node.Y + (maxOffY > 0 ? rng.Next(0, maxOffY + 1) : 0);

            node.RoomX = rx;
            node.RoomY = ry;
            node.RoomW = rw;
            node.RoomH = rh;

            for (int x = rx; x < rx + rw; x++)
                for (int y = ry; y < ry + rh; y++)
                    map.Tiles[x, y] = TileType.Floor;

            return;
        }

        if (node.Left  != null) CarveRooms(map, node.Left);
        if (node.Right != null) CarveRooms(map, node.Right);

        // Пробрасываем координаты комнаты вверх по дереву
        // (используем левый лист как «опорную» точку для коридора)
        var leftLeaf  = GetAnyLeaf(node.Left!);
        node.RoomX = leftLeaf.RoomX;
        node.RoomY = leftLeaf.RoomY;
        node.RoomW = leftLeaf.RoomW;
        node.RoomH = leftLeaf.RoomH;
    }

    // ── Коридоры: соединяем центры комнат Г-образно ───────────────────────────
    private void CarveCorridors(DungeonMap map, BspNode node)
    {
        if (node.IsLeaf) return;

        CarveCorridors(map, node.Left!);
        CarveCorridors(map, node.Right!);

        var leftLeaf  = GetAnyLeaf(node.Left!);
        var rightLeaf = GetAnyLeaf(node.Right!);

        int x1 = leftLeaf.RoomX  + leftLeaf.RoomW  / 2;
        int y1 = leftLeaf.RoomY  + leftLeaf.RoomH  / 2;
        int x2 = rightLeaf.RoomX + rightLeaf.RoomW / 2;
        int y2 = rightLeaf.RoomY + rightLeaf.RoomH / 2;

        // Случайно выбираем порядок Г-образного коридора
        if (rng.Next(2) == 0)
        {
            CarveHLine(map, x1, x2, y1);
            CarveVLine(map, y1, y2, x2);
        }
        else
        {
            CarveVLine(map, y1, y2, x1);
            CarveHLine(map, x1, x2, y2);
        }
    }

    private void CarveHLine(DungeonMap map, int x1, int x2, int y)
    {
        int lo = Math.Min(x1, x2);
        int hi = Math.Max(x1, x2);
        for (int x = lo; x <= hi; x++)
            TrySetFloor(map, x, y);
    }

    private void CarveVLine(DungeonMap map, int y1, int y2, int x)
    {
        int lo = Math.Min(y1, y2);
        int hi = Math.Max(y1, y2);
        for (int y = lo; y <= hi; y++)
            TrySetFloor(map, x, y);
    }

    private static void TrySetFloor(DungeonMap map, int x, int y)
    {
        if (x <= 0 || x >= map.Width - 1 || y <= 0 || y >= map.Height - 1) return;
        if (map.Tiles[x, y] == TileType.Wall)
            map.Tiles[x, y] = TileType.Floor;
    }

    // ── Ямы в комнатах: у стен и в центрах ───────────────────────────────────
    /// <summary>
    /// Размещает ямы двумя способами:
    ///   1. У стен внутри комнаты — Floor-тайлы, у которых есть кардинальный сосед-стена.
    ///      Вероятность: 35%.
    ///   2. В центральной зоне комнаты (отступ ≥2 от края) — случайные ямы.
    ///      Вероятность: 20%.
    /// Защитная зона 3 тайла вокруг Exit остаётся свободной.
    /// </summary>
    private void PlacePitsInRooms(DungeonMap map, List<BspNode> leaves, int exitX, int exitY)
    {
        const int ExitClearRadius = 3;   // тайлы вокруг Exit — без ям
        const float WallPitChance   = 0.45f;
        const float CenterPitChance = 0.33f;

        int[] cardDx = {  0,  0, -1, 1 };
        int[] cardDy = { -1,  1,  0, 0 };

        foreach (var leaf in leaves)
        {
            int rx = leaf.RoomX, ry = leaf.RoomY;
            int rw = leaf.RoomW, rh = leaf.RoomH;

            // 1. Ямы у стен — проверяем всю площадь комнаты
            for (int x = rx; x < rx + rw; x++)
            for (int y = ry; y < ry + rh; y++)
            {
                if (map.Tiles[x, y] != TileType.Floor) continue;
                if (NearExit(x, y, exitX, exitY, ExitClearRadius)) continue;

                // Есть хотя бы один кардинальный сосед — Wall?
                bool adjacentToWall = false;
                for (int d = 0; d < 4; d++)
                {
                    int nx = x + cardDx[d], ny = y + cardDy[d];
                    if (nx < 0 || nx >= map.Width || ny < 0 || ny >= map.Height) continue;
                    if (map.Tiles[nx, ny] == TileType.Wall) { adjacentToWall = true; break; }
                }
                if (!adjacentToWall) continue;

                if ((float)rng.NextDouble() < WallPitChance)
                    map.Tiles[x, y] = TileType.Pit;
            }

            // 2. Ямы в центре комнаты (отступ 2 тайла от края)
            for (int x = rx + 2; x < rx + rw - 2; x++)
            for (int y = ry + 2; y < ry + rh - 2; y++)
            {
                if (map.Tiles[x, y] != TileType.Floor) continue;
                if (NearExit(x, y, exitX, exitY, ExitClearRadius)) continue;

                if ((float)rng.NextDouble() < CenterPitChance)
                    map.Tiles[x, y] = TileType.Pit;
            }
        }
    }

    private static bool NearExit(int x, int y, int exitX, int exitY, int radius)
        => Math.Abs(x - exitX) + Math.Abs(y - exitY) <= radius;

    // ── BFS: спавн игрока на максимальном расстоянии от Exit ──────────────────
    private (int x, int y) FindMaxDistanceSpawn(DungeonMap map, int exitX, int exitY)
    {
        var dist = new int[map.Width, map.Height];
        for (int x = 0; x < map.Width; x++)
            for (int y = 0; y < map.Height; y++)
                dist[x, y] = -1;

        var queue = new Queue<(int, int)>();
        queue.Enqueue((exitX, exitY));
        dist[exitX, exitY] = 0;

        int bestX = exitX, bestY = exitY, maxDist = 0;

        int[] ddx = { 0,  0, -1, 1 };
        int[] ddy = { -1, 1,  0, 0 };

        while (queue.Count > 0)
        {
            var (cx, cy) = queue.Dequeue();

            for (int i = 0; i < 4; i++)
            {
                int nx = cx + ddx[i];
                int ny = cy + ddy[i];

                if (nx < 0 || nx >= map.Width || ny < 0 || ny >= map.Height) continue;
                if (dist[nx, ny] != -1) continue;

                // BFS для спавна — только Floor и Exit, без Pit
                if (!map.IsWalkable(nx, ny)) continue;

                dist[nx, ny] = dist[cx, cy] + 1;
                queue.Enqueue((nx, ny));

                if (dist[nx, ny] > maxDist)
                {
                    maxDist = dist[nx, ny];
                    bestX = nx;
                    bestY = ny;
                }
            }
        }

        return (bestX, bestY);
    }

    // ── Вспомогательные методы BSP-узлов ──────────────────────────────────────
    private static List<BspNode> CollectLeaves(BspNode root)
    {
        var list = new List<BspNode>();
        Traverse(root, list);
        return list;

        static void Traverse(BspNode n, List<BspNode> acc)
        {
            if (n.IsLeaf) { acc.Add(n); return; }
            if (n.Left  != null) Traverse(n.Left,  acc);
            if (n.Right != null) Traverse(n.Right, acc);
        }
    }

    private static BspNode GetAnyLeaf(BspNode node)
    {
        while (!node.IsLeaf)
            node = node.Left ?? node.Right!;
        return node;
    }
}

// ── Узел BSP-дерева ────────────────────────────────────────────────────────────
internal sealed class BspNode
{
    // Область секции в координатах карты
    public int X, Y, W, H;

    // Комната внутри секции (заполняется при CarveRooms)
    public int RoomX, RoomY, RoomW, RoomH;

    // Дочерние узлы
    public BspNode? Left;
    public BspNode? Right;

    public bool IsLeaf => Left == null && Right == null;

    public BspNode(int x, int y, int w, int h)
    {
        X = x; Y = y; W = w; H = h;
    }
}
