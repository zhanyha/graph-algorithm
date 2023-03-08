图论算法

## 一、并查集

参考本人博客：[做最好的博客_CSDN博客](https://blog.csdn.net/idefined?type=blog)

## 1.介绍：

在计算机科学中，并查集是一种树型的数据结构，用于处理一些不交集（`Disjoint Sets`）的合并及查询问题。有一个联合-查找算法（`Union-find Algorithm`）定义了两个用于此数据结构的操作：

`Find`：确定元素属于哪一个子集。它可以被用来确定两个元素是否属于同一子集。
`Union`：将两个子集合并成同一个集合。
由于支持这两种操作，一个不相交集也常被称为联合-查找数据结构（`Union-find Data Structure`）或合并-查找集合（`Merge-find Set`）。
为了更加精确的定义这些方法，需要定义如何表示集合。一种常用的策略是为每个集合选定一个固定的元素，称为代表，以表示整个集合。接着，`Find(x)` 返回 `x` 所属集合的代表，而 `Union` 使用两个集合的代表作为参数

**并查集在图论中的一些应用：**

> 1. 并查集动态环检测
>
> 2. 并查集检测连通性

## 2. 快速查找的并查集（`quick find`）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200708194544707.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lkZWZpbmVk,size_16,color_FFFFFF,t_70)
元素的id值一样，（这里是`0 1`）就代表他们在一个集合里面
显然查找id的时间复杂度是`O(1)`的，而在合并的时候需要遍历整个数组在这里是找到元素值为`0/1`的元素，修改成`1/0`。也就是将要合并元素的其中一个`id`换成另一个元素的`id`,具体看代码，很简单的

```cpp
class UnionFind {
private:
    int* id;
    int count;  //并查集中元素的个数
public:
    UnionFind(int count){
        this->count = count;
        id = new int[count];
        for (int i = 0; i < count; ++i)
            id[i] = i;
    }
    ~UnionFind(){
        delete[] id;
    }
    int find(int p){
        assert(p>=0 && p < count);
        return id[p];
    }
    bool isConnected(int p,int q){
        return find(p) == find(q);
    }
    void unionElements(int p,int q){
        int pID = find(p);
        int qID = find(q);
        if(pID == qID)
            return;
        for (int i = 0; i < count; ++i)
            if(id[i] == pID)
                id[i] = qID;
    }
};
```
## 3. 快速合并的并查集(`quick union`)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200708214857337.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lkZWZpbmVk,size_16,color_FFFFFF,t_70)

这次`parent`数组中记录的是他的==父亲节点==，初始的时候，我们让`parent`数组中的父亲节点都指向自己，也就是图中的环。

**查找:** 我们元素所在集合的时候都是去查找该元素的根节点，怎么找到他的根节点呢? 就是不断的往上查找父亲节点的过程，直到某个节点的父亲节点指向自己。

**合并：** 我们每次让其中一个元素的跟节点指向另一个元素的根节点就好了，所以合并的过程是很快的。
```cpp
class UnionFind {
private:
    int* parent;
    int count;  //元素的个数
public:
    UnionFind(int count){
        this->count = count;
        parent = new int[count];
        for (int i = 0; i < count; ++i)
            parent[i] = i;
    }
    ~UnionFind(){
        delete[] parent;
    }
    int find(int p){
        assert(p>=0 && p < count);
        while(p != parent[p])
            p = parent[p];
        return p;
    }
    bool isConnected(int p,int q){
        return find(p) == find(q);
    }
    void unionElements(int p,int q){
        int pRoot = find(p);
        int qRoot = find(q);
        if(pRoot == qRoot)
            return;
        parent[pRoot] = qRoot;
    }
};
```

## 4. 再次优化
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200708201446198.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lkZWZpbmVk,size_16,color_FFFFFF,t_70)
考虑一下，合并`2` 和`4`进行合并操作。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200708201604672.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lkZWZpbmVk,size_16,color_FFFFFF,t_70)
我们每次将高度更小的树，合并到另一颗树中。这样树的高度就没有那么高，那么当数据量特别大的时候，查找到的速度也会得到有效的提高。

`rank[i]` : 设定一个`rank`数组，`rank`数组的意思是根节点为`i`的树的高度。
而且想一想，只有当两个要合并的元素的根的`rank`值，也就是高度是一致的时候，我们才需要去维护`rank`数组。
**代码：**

```cpp
class UnionFind {
private:
    int* parent;
    int* rank;
    int count;  //并查集中元素的个数
public:
    UnionFind(int count){
        this->count = count;
        parent = new int[count];
        rank = new int[count];
        for (int i = 0; i < count; ++i) {
            parent[i] = i;
            rank[i] = 1;//初始时，高度为1，设置成0也是没问题的
        }
    }
    ~UnionFind(){
        delete[] parent;
        delete[] rank;
    }
    int find(int p){
        assert(p>=0 && p < count);
        while(p != parent[p])
            p = parent[p];
        return p;
    }
    bool isConnected(int p,int q){
        return find(p) == find(q);
    }
    void unionElements(int p,int q){
        int pRoot = find(p);
        int qRoot = find(q);
        if(pRoot == qRoot)
            return;
        if(rank[pRoot] < rank[qRoot])
            parent[pRoot] = qRoot;
        else if(rank[pRoot] > rank[qRoot])
            parent[qRoot] = pRoot;
        else {//rank[pRoot] == rank[qRoot]
            parent[pRoot] = qRoot;
            rank[qRoot]++;
        }
    }
};
```
现在的处理速度已经非常可观了。

## 5. 路径压缩
**思想：** 我们每次查找的时候，将当前节点指向父亲节点的父亲节点。这样每次查找的时候，我们都将高度在减小，这就是路径压缩。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200708214230315.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lkZWZpbmVk,size_16,color_FFFFFF,t_70)
第一步：![在这里插入图片描述](https://img-blog.csdnimg.cn/20200708214323386.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lkZWZpbmVk,size_16,color_FFFFFF,t_70)
第二步![在这里插入图片描述](https://img-blog.csdnimg.cn/20200708214351489.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lkZWZpbmVk,size_16,color_FFFFFF,t_70)
想一想其实这并不影响我们并查集的实现。因为，我们的目的就是为了找到某个元素的根节点，两个元素的根节点相同与否就决定了这两个元素是否在同一个集合当中。但是，当我们将路径压缩之后，我们树的高度进一步减小，那么查找的速度就会再次提高很多。整体的性能也会大大提升。


```cpp
class UnionFind {
private:
    int* parent;
    int* rank;
    int count;  //并查集中元素的个数
public:
    UnionFind(int count){
        this->count = count;
        parent = new int[count];
        rank = new int[count];
        for (int i = 0; i < count; ++i) {
            parent[i] = i;
            rank[i] = 1;//初始时，高度为1，设置成0也是没问题的
        }
    }
    ~UnionFind(){
        delete[] parent;
        delete[] rank;
    }
    int find(int p){
        assert(p>=0 && p < count);
        while(p != parent[p]) {
            p = parent[parent[p]];
            p = parent[p];
        }
        return p;
    }
    bool isConnected(int p,int q){
        return find(p) == find(q);
    }
    void unionElements(int p,int q){
        int pRoot = find(p);
        int qRoot = find(q);
        if(pRoot == qRoot)
            return;
        if(rank[pRoot] < rank[qRoot])
            parent[pRoot] = qRoot;
        else if(rank[pRoot] > rank[qRoot])
            parent[qRoot] = pRoot;
        else {//rank[pRoot] == rank[qRoot]
            parent[pRoot] = qRoot;
            rank[qRoot]++;
        }
    }
};
```

尝试一道能利用并查集解决的问题 ： **[朋友圈](https://leetcode-cn.com/problems/friend-circles/)**
我的参考代码：

```cpp
class Solution {
private:
    int* parent;
    int* rank;
    int count;
public:
    int find(int p){
        while(p!= parent[p]){
            parent[p] = parent[parent[p]];
            p = parent[p];
        }
        return p;
    }
    void unionElements(int q,int p){
        int pRoot = find(p);
        int qRoot = find(q);
        if(qRoot == pRoot) return;
        if(rank[qRoot] < rank[pRoot]){
            parent[qRoot] = pRoot;
        } else if(rank[qRoot] > rank[pRoot]){
            parent[pRoot] = qRoot;
        }else{
            parent[pRoot] = qRoot;
            rank[pRoot]++;
        }
    }
    int findCircleNum(vector<vector<int>>& M) {
        int n = M.size();
        this->count = n;
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < count; ++i) {
            parent[i] = i;
            rank[i] = 1;//树的高度
        }
        for (int i = 0; i < n; ++i)
            for (int j = i+1; j < n; ++j) {
                if(M[i][j] == 1)
                    unionElements(i,j);
            }
        vector<int> v = vector<int>(count,-1);
        for (int k = 0; k < count; ++k) {
            v[find(parent[k])] = 1;
        }
        int res = 0;
        for (int l = 0; l < count; ++l) {
            if(v[l] == 1)
                res++;
        }
        return res;
    }
    ~Solution(){
        delete [] rank;
        delete [] parent;
    }
};
```

# 二、图的存储方式

	1.邻接矩阵
	2.邻接表
	3.边集
# 三、无向图的遍历

## 1 Depth First Search

存储方式为邻接矩阵



时间复杂度: O(V+E)

#### 连通图

#### 非连通图

思路

> 一行代码：在对没有访问的节点再次进行dfs

#### 相关问题

1 求连通分量的个数以及每个连通分量的顶点个数

思路
每次dfs用不同的list存储， 结果在visit数据中
2 单源路径问题
求一个顶点到可以连通的其他点的路径
思路
dfs加一个parent参数
3 判断从一个点能否到达另一个顶点，单源路径的提前结束
思路
把dfs的返回值该为boolean,找到了就直接返回true
4 无向图的环检测
5 二分图的检测

## 2 Breadth First Search

存储方式为邻接表

时间复杂度
O(V+E)

#### 连通图

#### 非连通图

思路

>  一行代码：在对没有访问的节点再次进行bfs

#### 相关问题

1 单元路径问题

2 利用bfs实现的无权图的路径最短问题的方法
3 求最短路径的距离

## 3 DFS和BFS的关系

DFS的非递归模式的算法模型和BFS一摸一样，只是用的容器不一样，一个是栈Stack,一个是队列



### leecode上DFS与BFS相关问题

**floodfill问题**
733
图像渲染
1034
边框着色
200
岛屿的数量
1020
飞地的数量
130
被围绕的区域
529
扫雷游戏
827
hard 最大人工岛屿
**BFS**
1091
二进制矩阵中的最短路径
**图的建模核心 状态转移**
752
打开转盘锁
**一道智力题**
如何使用一个5升桶，一个3升桶来装出4升水
**又一道智力题**
狼羊菜人渡河问题，船比较小，如何安全渡河。就是狼不能和羊同时在一边，羊和菜不能同时在一边
**leetcode hard 773**
滑动谜题



# 四、最小生成树问题

## 1 生成树

一个连通图的生成树是指一个连通子图，它含有图中全部n个顶点，但只有足以构成一棵树的n-1条边。一颗有n个顶点的生成树有且仅有n-1条边，如果生成树中再添加一条边，则必定成环。

## 2 最小生成树

在连通网的所有生成树中，所有边的代价和最小的生成树，称为最小生成树。

### 1.Kruskal算法

此算法可以称为“加边法”，初始最小生成树边数为0，每迭代一次就选择一条满足条件的最小代价边，加入到最小生成树的边集合里。

#### 算法流程

```
1 把图中的所有边按代价从小到大排序； 

2 把图中的n个顶点看成独立的n棵树组成的森林； 

3 按权值从小到大选择边，所选的边连接的两个顶点ui,vi,应属于两颗不同的树，则成为最小生成树的一条边，并将这两颗树合并作为一颗树。 

4 重复(3),直到所有顶点都在一颗树内或者有n-1条边为止。
```

### 2.Prim算法

此算法可以称为“加点法”，每次迭代选择代价最小的边对应的点，加入到最小生成树中。算法从某一个顶点s开始，逐渐长大覆盖整个连通网的所有顶点。

```
1.图的所有顶点集合为V；初始令集合u={s},v=V−u;
2.在两个集合u,v能够组成的边中，选择一条代价最小的边(u0,v0)，加入到最小生成树中，并把v0并入到集合u中。
3.重复上述步骤，直到最小生成树有n-1条边或者n个顶点为止。
注意：由于不断向集合u中加点，所以最小代价边必须同步更新；需要建立一个辅助数组closedge,用来维护集合v中每个顶点与集合u中最小代价边信息
```

### 实际应用

网络建设，修桥，修路
比如把所有村庄用桥连通。要求费用最少



# 五、最短路径算法

## 1 Dijkstra算法

Dijkstra算法采用的是一种贪心的策略，声明一个数组dis来保存源点到各个顶点的最短距离和一个保存已经找到了最短路径的顶点的集合：T，初始时，原点 s 的路径权重被赋为 0 （dis[s] = 0）。若对于顶点 s 存在能直接到达的边（s,m），则把dis[m]设为w（s, m）,同时把所有其他（s不能直接到达的）顶点的路径长度设为无穷大。初始时，集合T只有顶点s。



然后，从dis数组选择最小值，则该值就是源点s到该值对应的顶点的最短路径，并且把该点加入到T中，OK，此时完成一个顶点，



再然后，我们需要看看新加入的顶点是否可以到达其他顶点并且看看通过该顶点到达其他点的路径长度是否比源点直接到达短，如果是，那么就替换这些顶点在dis中的值。



最后，又从dis中找出最小值，重复上述动作，直到T中包含了图的所有顶点。
*复杂度O(V²)*
**注意**

```
不能用来求有负权边的最短路径
因为使用Integer.MAXVALUE代表无穷大，所以可能存在溢出
Dijkstra在求从源点s到终点t的最短距离时，是可以提前结束的
```

## 2 Dijkstra算法使用优先队列的优化

实现代码注意事项

```
每次修改的dis数组中的值并不会更新优先队列中的值，所以要重新加一个Node进去，每次取最小值的时候需要判断这个顶点有没有被访问过。

复杂度O(ElogE)
```



## 3 Dijkstra算法记录路径

## 4 bellman-ford算法

1. 初始化时将起点s到各个顶点v的距离dist(s->v)赋值为∞，dist(s->s)赋值为0



2. 后续进行最多n-1次遍历操作(n为顶点个数,上标的v输入法打不出来...),对所有的边进行松弛操作,假设:                                                                                所谓的松弛，以边ab为例，若dist(a)代表起点s到达a点所需要花费的总数，
   dist(b)代表起点s到达b点所需要花费的总数,weight(ab)代表边ab的权重，
   若存在:
   (dist(a) +weight(ab)) < dist(b)
   则说明存在到b的更短的路径,s->...->a->b,更新b点的总花费为(dist(a) +weight(ab))，父节点为a



3. 如果各点没有更新了，说明以求出结果。所以可以用一个标志位来记录

### 4.1 bellman-ford的算法核心

每经过一轮松弛操作，就找到了多经过一条边到达某点的距离最短，所以最多经过V-1轮

**实现代码的含义**

​	dis[v]的语义是 从源点s到顶点v最多经过k条边的最短路径
**注意**：

最多进行n-1次遍历操作。如果再来一次还能更新。说明存在负环路。因为最多一个顶点经过所有顶点也就是进行n-1次松弛求得最短路径

### 4.2 总结

**复杂度O（V*E）**
本质是个有向图算法
bellman-Ford在处理无向图的时候，如果有一条边是负权的话，就一定存在负权环。
bellman-ford是可以处理有向图存在负权的情况的。
bellman-Ford在求从源点s到终点t的最短距离时，是不能提前结束的

## bellman-ford算法记录路径

## SPFA

​	是 `bellman-ford`算法的优化

## Folyd算法

思想

通过`Floyd`计算图`G=(V,E)`中各个顶点的最短路径时，需要引入两个矩阵，矩阵`S`中的元素`a[i][j]`表示顶点i(第i个顶点)到顶点j(第j个顶点)的距离。矩阵P中的元素`b[i][j]`，表示顶点i到顶点j经过了`b[i][j]`记录的值所表示的顶点。

**算法过程**

假设图`G`中顶点个数为`N`，则需要对矩阵`D`和矩阵`P`进行N次更新。初始时，矩阵`D`中顶点`a[i][j]`的距离为顶点i到顶点`j`的权值；如果`i`和j不相邻，则`a[i][j]=∞`，矩阵`P`的值为顶点`b[i][j]`的`j`的值。 接下来开始，对矩阵`D`进行`N`次更新。第`1`次更新时，如果”`a[i][j]`的距离” > “`a[i][0]+a[0][j]`”(`a[i][0]+a[0][j]`表示”`i与j`之间经过第1个顶点的距离”)，则更新`a[i][j]为”a[i][0]+a[0][j]”`,更新`b[i][j]=b[i][0]`。 同理，第`k`次更新时，如果`”a[i][j]的距离” > “a[i][k-1]+a[k-1][j]”`，则更新`a[i][j]为”a[i][k-1]+a[k-1][j]”,b[i][j]=b[i][k-1]`。更新`N`次之后，操作完成！
应用：求图的直径
时间复杂度`O(V*V*V)`

## 对比

可以用来计算权值存在负数的情况
只能计算从一个顶点到另一个顶点的最短路径，也就是局部最短，并且不适用于存在负数权的图
`Folyd`算法是解决任意两点间的最短路径的一种算法，可以正确处理无向图或有向图或负权（但不可存在负权回路)的最短路径问题，同时也被用于计算有向图的传递闭包。



















# 六、桥与割点

## 1 桥

对于无向图，如果删除一条边，整图的联通数量发生了变化，那么就称这条边为桥
实现思想
**low[w]>ord[v]就一定是一个桥**

## 2 DFS遍历树和BFS遍历树

​	dfs遍历树能够回到祖先节点
​	bfs只能回到兄弟节点

## 3 割点

对于无向图，如果删除一个顶点，整图的联通数量发生了变化，那么就称这个点为割点

实现细节

**注意 ：**因为根节点的low[w]值一定小于等于ord[v]。所以要对根节点做特殊的判断。只需要判断根节点是否有超过一个孩子。基于遍历树去判断孩子个数



# 七、哈密尔顿问题与欧拉问题

哈密尔顿回路与欧拉回路没有必然联系

## 1 哈密尔顿回路

从一个点出发，每个点只走一次，将每个点都走一遍，再回到出发点



从任一顶点 暴力搜索+回溯  的实现算法



## 2 哈密尔顿路径

**leetcode 980**

## 3 欧拉回路

1 如果没有平行边的无向图，那么如果连通分量为1，并且每个点的度为偶数就一定有欧拉回路
2 对于有向图，欧拉道路存在的充要条件是：最多只能有两个点的入度不等于出度，额日期额必须是其中一个点的出度恰好比入读大1（起点），另一个的入度比出度大1（终点）

### 3.1 三种实现算法

1 回溯法
时间复杂度O(n!)
2 Fleury算法

如果又多条边可以走，选择不是桥的边走，因为如果走了桥，那桥这条边一旦删除，一定把图分割成了两部分
到一个顶点，对每邻边判断是否是桥

时间复杂度O(E²)
3 Hierholzer算法

使用两个栈，算法思想就是把证明欧拉回路的过程实现一遍



## 4 欧拉路径

如果是一个无向连通图，且连通图存在平行边。且最多只存在2个奇点(度数为奇数的)，则一定存在欧拉道路

如果有两个奇点，那么他们一定是起点和终点

如果奇点不存在，可以从任意点出发，最终一定会回到该点，这称为欧拉回路

Hierholzer算法

​	只不过起始点不能乱选
