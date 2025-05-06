# %%
from nebula_graph import NebulaGraph

ng = NebulaGraph()
# %%
ng.execute(
    """
    MATCH ()<-[e:follow]-() 
    RETURN e 
    LIMIT 3;
    """
)
# %%
ng.execute(
    """
    MATCH ()<-[e:serve]-() 
    RETURN e.start_year AS StartYear, e.end_year AS EndYear
    LIMIT 3;
    """
)
# %%
ng.execute(
    """
    MATCH (v1:player)-[e:serve]->(v2:team) 
    RETURN v2.team.name, v1.player.name, e.start_year AS StartYear, e.end_year AS EndYear
    LIMIT 3;
    """
)
# %%
ng.execute(
    """
    MATCH (v) 
    WHERE id(v) IN ['player101', 'player102']
    RETURN v;
    """
)
# %%
ng.execute(
    """
    MATCH (v) 
    WHERE id(v) == 'player101' 
    RETURN v;
    """
)
# %%
ng.execute(
    """
    MATCH (v:player)
    RETURN v
    LIMIT 3;
    """
)
# %%
ng.execute(
    """
    MATCH (v:team)
    RETURN v
    """
)
# %%
ng.execute(
    """
    MATCH (v:player)
    WHERE v.player.name == "Tim Duncan"
    RETURN v
    """
)
# %%
ng.execute(
    """
    MATCH (v:player{name:"Tim Duncan"})
    RETURN v
    """
)

# %%
ng.execute(
    """
    MATCH (v) WHERE id(v) IN ['player100', 'player101']
    RETURN v.player.name AS name;
    """
)
# %%
ng.execute(
    """
    MATCH (v:player{name:"Tim Duncan"})-[e:follow{degree:90}]->(v2)
    RETURN e
    """
)
# %%
ng.execute(
    """
    MATCH (v1:player)-->(v2:player) 
    WHERE v1.player.name in ['Tim Duncan', 'Yao Ming'] 
    RETURN v1, v2;
    """
)
# edge from player to player is follow
# Tim Duncan follows [Tony Parker] and [Manu Ginobili]
# Yao Ming follows [Tracy McGrady] and [Shaquille O'Neal]
# %%
ng.execute(
    """
    MATCH (v1:player)-->(v2:team) 
    WHERE v1.player.name == 'Chris Paul'
    RETURN v1, v2;
    """
)
# edge from player to team is serve
# Chris Paul serves [Rockets], [Clippers], [Hornets]
# %%
ng.execute(
    """
    MATCH (v1:player)-->(v2:team)<--(v3:player) 
    WHERE v1.player.name == 'Tim Duncan'
    RETURN v2.team.name AS Team, v3.player.name AS Name;
    """
)

# %%
ng.execute(
    """
    MATCH p=(v1:player)-->(v2:player) 
    WHERE v1.player.name == 'Tim Duncan' 
    RETURN p;
    """
)
# %%
ng.execute(
    """
    MATCH p=(v1:player)-[e:follow]->(v2:player) 
    WHERE v1.player.name == 'Tim Duncan' 
    RETURN p;
    """
)

# %%
# 找出特定节点 m（ID为 "player100" 的节点）直接连接的节点，以及这些节点 n 直连的下一层节点 l (OPTIONAL MATCH (n)-[]->(l))， 如果没有找到这样的节点 l，查询也会继续， l 的返回值是 null。
ng.execute(
    """
    MATCH (m)-[]->(n) WHERE id(m)=="player100" 
    OPTIONAL MATCH (n)-[]->(l) 
    RETURN id(m),id(n),id(l);
    """
)

# %%
