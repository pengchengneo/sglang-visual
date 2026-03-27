export interface RadixNode {
  id: string;
  tokens: string[];
  refCount: number;
  children: Map<string, RadixNode>;
  parent: RadixNode | null;
  state: "active" | "cached" | "evictable" | "inserting" | "evicting" | "matching";
}

export interface RadixTree {
  root: RadixNode;
  blockCount: number;
  maxBlocks: number;
}

let nodeIdCounter = 0;

export function createNode(tokens: string[], parent: RadixNode | null): RadixNode {
  return { id: `node-${nodeIdCounter++}`, tokens, refCount: 0, children: new Map(), parent, state: "cached" };
}

export function createTree(maxBlocks: number): RadixTree {
  nodeIdCounter = 0;
  return { root: createNode(["<root>"], null), blockCount: 1, maxBlocks };
}

export function findPrefix(tree: RadixTree, tokens: string[]): { matchedNodes: RadixNode[]; matchedLength: number } {
  const matchedNodes: RadixNode[] = [tree.root];
  let pos = 0;
  let current = tree.root;
  while (pos < tokens.length) {
    const nextToken = tokens[pos];
    const child = current.children.get(nextToken);
    if (!child) break;
    let i = 0;
    while (i < child.tokens.length && pos + i < tokens.length && child.tokens[i] === tokens[pos + i]) { i++; }
    if (i === child.tokens.length) {
      matchedNodes.push(child);
      pos += child.tokens.length;
      current = child;
    } else { break; }
  }
  return { matchedNodes, matchedLength: pos };
}

export function insertSequence(tree: RadixTree, tokens: string[]): RadixNode {
  const { matchedNodes, matchedLength } = findPrefix(tree, tokens);
  let current = matchedNodes[matchedNodes.length - 1];
  let pos = matchedLength;
  while (pos < tokens.length) {
    const chunkSize = Math.min(4, tokens.length - pos);
    const chunk = tokens.slice(pos, pos + chunkSize);
    const newNode = createNode(chunk, current);
    newNode.state = "inserting";
    current.children.set(chunk[0], newNode);
    tree.blockCount++;
    current = newNode;
    pos += chunkSize;
  }
  for (const node of matchedNodes) { node.refCount++; node.state = "active"; }
  let n: RadixNode | null = current;
  while (n && !matchedNodes.includes(n)) { n.refCount++; n.state = "active"; n = n.parent; }
  return current;
}

export function releaseSequence(_tree: RadixTree, leafNode: RadixNode): void {
  let node: RadixNode | null = leafNode;
  while (node) {
    node.refCount = Math.max(0, node.refCount - 1);
    node.state = node.refCount > 0 ? "active" : "evictable";
    node = node.parent;
  }
}

export function evictNodes(tree: RadixTree, count: number): RadixNode[] {
  const evicted: RadixNode[] = [];
  function findEvictableLeaves(node: RadixNode): RadixNode[] {
    if (node.children.size === 0 && node.refCount === 0 && node !== tree.root) return [node];
    const leaves: RadixNode[] = [];
    for (const child of node.children.values()) leaves.push(...findEvictableLeaves(child));
    return leaves;
  }
  for (let i = 0; i < count; i++) {
    const leaves = findEvictableLeaves(tree.root);
    if (leaves.length === 0) break;
    const victim = leaves[0];
    victim.state = "evicting";
    evicted.push(victim);
    if (victim.parent) {
      for (const [key, child] of victim.parent.children) {
        if (child === victim) { victim.parent.children.delete(key); break; }
      }
    }
    tree.blockCount--;
  }
  return evicted;
}

export interface FlatNode {
  node: RadixNode;
  depth: number;
  parentId: string | null;
  childIndex: number;
  totalSiblings: number;
}

export function flattenTree(tree: RadixTree): FlatNode[] {
  const result: FlatNode[] = [];
  function walk(node: RadixNode, depth: number, childIndex: number, totalSiblings: number) {
    result.push({ node, depth, parentId: node.parent?.id ?? null, childIndex, totalSiblings });
    const children = Array.from(node.children.values());
    children.forEach((child, i) => walk(child, depth + 1, i, children.length));
  }
  walk(tree.root, 0, 0, 1);
  return result;
}

export function analyzeRequest(tree: RadixTree, tokens: string[]): { hits: number; misses: number; hitRatio: number } {
  const { matchedLength } = findPrefix(tree, tokens);
  const hits = matchedLength;
  const misses = tokens.length - matchedLength;
  const hitRatio = tokens.length > 0 ? hits / tokens.length : 0;
  return { hits, misses, hitRatio };
}
