#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import operator
import numpy as np
import networkx as nx
import math
from scipy.sparse import *
import csv

##Parallelisation
import petsc4py
petsc4py.init()
import slepc4py
slepc4py.init()
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI

c_petsc = PETSc.COMM_WORLD.duplicate()
c_mpi4py = c_petsc.tompi4py()

########################################################################
#MAIN
########################################################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('input_file_name')
    parser.add_argument('output_file_name')
    parser.add_argument('version',type=int)
    parser.add_argument('--teleport','-d',type=float)
    parser.add_argument('--laziness','-q',type=float)
    parser.add_argument('--verbose',type=bool)

    args = parser.parse_args()

    filepath = args.input_file_name
    version = args.version
    output = args.output_file_name

    if args.teleport:
        d=args.teleport
    else:
        d=0.01

    if args.laziness:
        q=args.laziness
    else:
        q=0

    # Read the mvtfile into a dictionary of dictionaries representing successive
    # time-slices.
    if verbose:
        print("Reading mvtfile...")
    dicosnap, edgelist = readcsv(filepath)
    if verbose:
        print("Beginning: " + str(min(dicosnap.keys())))
        print("End: " + str(max(dicosnap.keys())))
        print("Number of time-slices: " + str(len(dicosnap)))

    # Build the time-aggregated graph
    Gccm, dicoedgeccm = build_graph(edgelist,d)

    # Index nodes by integers from 0 to n-1
    correspnodes = {}
    listnodes = sorted(Gccm.nodes())
    nb_nodes = len(listnodes)
    print("Number of nodes: " + str(nb_nodes),flush=True)
    correspnodes = {listnodes[i]: i for i in range(nb_nodes)}

    # Compute TempoRank centrality
    dicosnapccm = gccdicosnap(dicosnap,dicoedgeccm)
    listtrans,listod = listetransition(dicosnapccm,correspnodes,nb_nodes,q,d,
        version)
    Ptp = multparpaquet(listtrans)
    V1 = slep_powerit(Ptp,nb_nodes)
    Ptp.destroy()
    TR = temporankv2(V1,listtrans,listnodes,listod,version)
    V1.destroy()
    Print("Fin TR",flush=True)


    ##Ordonner les scores TR
    rank_orderi = [key for (key, value) in sorted(TR.items(), key=operator.itemgetter(1), reverse=True)]
    ordered_tri = np.array([TR[k] for k in rank_orderi])

    static_order = [key for (key, value) in sorted(pr_basicccm.items(), key=operator.itemgetter(1), reverse=True)]
    ordered_spr = np.array([pr_basicccm[k] for k in static_order])
    #ordre temporal
    sorted_SPR = np.array([pr_basicccm[k] for k in rank_orderi])


    ##Sortie en fichier csv
    Csv = csv.writer(open(output, "w"))
    Csv.writerow(["%s|%s|%s"%("Node","TempoRankccm","Static_PRccm")])
    for i in range(len(rank_orderi)):
        Csv.writerow(["%s|%e|%e"%(rank_orderi[i],ordered_tri[i],sorted_SPR[i])])


    return 0

########################################################################
#Parser
########################################################################
def readcsv(infile):
    """
    Reads the mvtfile and returns the temporal network.

    Parameters:
        infile (string): Path to the mvtfile, formatted as a csv file with lines
        as date,source,destination.

    Returns:
        dicosnap (dict): Keys are the dates found in the mvtfile, values are
        dictionaries {edgelist:[edgelist]}, where edgelist is a list of tuples
        (source,destination).

        edgelist (list): List of all tuples (source,destination) that can be used
        in static analysis.
    """

    dicomvt = {} # Dictionary with a "time" key as well as numbered keys whose
    # values are sets of pairs (in the unweighted case) or lists of pairs (in
    # the weighted case)
    dicomvt["time"] = [] # Will contain all dates present in infile

    fd = open(infile,'r')
    header = fd.readline()

    for line in fd.readlines():
        line = line.strip()
        items = line.split(',')

        # Entries are formatted as tstamp,source,dest
        # timestamps are formatted as YYYY-MM-DD
        tstamp = items[0]
        source = items[1]
        dest = items[2]

        tstamp = datetime.strptime(tstamp, '%Y-%m-%d')

        if not tstamp in dicomvt["time"]:
            dicomvt["time"].append(tstamp)
            # We use a set in the unweighted case
            dicomvt[tstamp] = set()
        dicomvt[tstamp].add((source,dest))

    fd.close()

    ###############
    ##Snaps
    ###############
    ## Choisir des periodes d'activites specifiques!!
    debut = datetime(year=2005, month=1, day=1)
    resolution = timedelta(days=1)
    fin = datetime(year=2005, month=1, day=19)
    nbsnap = int(((fin-debut)+timedelta(days=1)).total_seconds()/resolution.total_seconds())

    datesnap = []
    for t in range(nbsnap):
        datesnap.append(debut+(resolution*(t+1))-timedelta(days=1))
    if datesnap[-1] < fin:
        datesnap.append(fin)

    dicosnap = {} # keys are the snapshot dates, values are dictionaries with a
    # single key "edgelist"
    edgelist = [] # Will accumulate directed edges
    for i in dicomvt["time"]:
        if i>=debut and i<=datesnap[-1]:
            flag = False
            j=0
            while not flag:
                if i <= datesnap[j]:
                    if datesnap[j] not in dicosnap.keys():
                        dicosnap[datesnap[j]] = {}
                        dicosnap[datesnap[j]]["edgelist"] = []
                    dicosnap[datesnap[j]]["edgelist"].extend(dicomvt[i])
                    edgelist.extend(dicomvt[i])
                    flag = True
                j += 1



    return dicosnap, edgelist


########################################################################
#Static (+CCM)
########################################################################

# Très très sale.

def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)

def getGraph(edgelist):
    """
    Builds the time-aggregated from the ground up, starting from the edgelist

    Parameters:
        edgelist (list): List of edges, formatted as tuples (source,destination)

    Returns:
        G (nx.DiGraph): Time-aggregated network described by edgelist, with an
        additional 'weight' attribute on the edges
        edges (dict): Dictionary indexed by all edges, whose values are weights
    """
    G = nx.DiGraph()
    edges = {}

    for edge in edgelist:
        edges[edge] = edges.get(edge, 0.0) + 1.0

    G.add_edges_from([(k[0],k[1], {'weight': v}) for k,v in edges.items()])

    return G, edges

def normweight(G,weights='real'):
    """
    Normalizes the "weight" attributes of edges of G by their total sum.

    Parameters:
        G (nx.DiGraph): A directed graph with a "weight" attribute on edges.
        weights (string): Chooses which weights are placed on edges:
            - 'random': Uniformly chosen in (0,1)
            - 'uniform': Each edge has equal weight
            - 'real': Normalizes existing weights

    Returns:
        G (nx.DiGraph): A directed graph such that the sum of all "weight"
        attributes on edges is equal to 1.
    """
    if weights == 'random':
        w = np.random.uniform(1e-5, 1.0, G.number_of_edges())
        w /= sum(w)
        c = 0
        for i in list(G.edges()):
            G[i[0]][i[1]]['weight'] = w[c]
            c += 1
    elif weights == 'uniform':
        w = 1.0/G.number_of_edges()
        for i in list(G.edges()):
            G[i[0]][i[1]]['weight'] = w
    else:
        out_degrees = [val for (node, val) in G.out_degree(weight='weight')]
        nrm = float(sum(out_degrees))
        for i in list(G.edges(data=True)):
            G[i[0]][i[1]]['weight'] = i[-1]['weight']/nrm

    return G

def undirtodirected(edges,gccedges):
    """
    Returns the directed subgraph containing all edges in gccedges

    Parameters:
        edges (dict): Dictionary of directed edges in G
        gccedges (nx.OutEdgeView): View of the edges of the undirected maximal
        connected component of G

    Returns:
        Gccm (nx.DiGraph): The maximal connected component as a directed graph
        dicoedgeccm (dict): Dictionary indexed by the edges of Gccm
    """
    Gccm = nx.DiGraph()
    dicoedgeccm ={}
    for edge in gccedges:
        if edge in edges.keys():
            dicoedgeccm[edge]=None # Placeholder value
            Gccm.add_edge(edge[0],edge[1], weight= edges[edge])

    #Retourne la composante connexe maximale agregee connexe ponderee
    return Gccm, dicoedgeccm

def build_graph(edgelist,d):
    """
    Constructs a directed graph from its edgelist.

    Parameters:
        edgelist (list): List of edges, formatted as tuples (source,destination)
        d (float): Teleportation parameter

    Returns:
        Gccm (nx.DiGraph): Depending on d, directed graph described by edgelist.
            - If d=0, returns the maximal (undirected) connected component of G,
            with correct directions added on its edges.
            - If d>0, returns the entire graph.
            In both cases, Gccm will have a 'weight' attribute on edges, with
            total sum 1.
    """
    G, edges = getGraph(edgelist) # Time-aggregated graph

    if d==0:
        # Select the largest (undirected) connected component of G
        Gcc = sorted(connected_component_subgraphs(G.to_undirected()),
            key = len, reverse=True)
        Gccmax = Gcc[0]
        gccedges = Gccmax.to_directed().edges()

        # Converts the connected component to a directed graph
        Gccmaxdir, dicoedgeccm = undirtodirected(edges,gccedges)

        Gccm = normweight(Gccmaxdir)
    else:
        Gccmaxdir, dicoedgeccm = undirtodirected(edges,edges)
        Gccm = normweight(Gccmaxdir)

    return Gccm, dicoedgeccm


########################################################################
#PETSC TempoRank
########################################################################
def gccdicosnap(dicosnap,dicoedgeccm):
    """
    Selects all time-stamped edges with nodes belonging to the maximal connected
    component.

    Parameters:
        dicosnap (dict): Dictionary indexed by timestamps, whose keys are
        edgelists for a given time-slice.
        dicoedgeccm (dict): Dictionary indexed by all edges in the aggregated
        network.

    Returns:
        dicosnapccm (dict): Dictionary indexed by timestamps, whose keys are
        edgelists for a given time-slice, with only edges in the c.c.
    """
    dicosnapccm = {}
    for t in dicosnap.keys():
        dicosnapccm[t] = {}
        dicosnapccm[t]["edgelist"] = []
        for edge in dicosnap[t]["edgelist"]:
            if edge in dicoedgeccm.keys():
                dicosnapccm[t]["edgelist"].append(edge)

    return dicosnapccm


#Remplissage de la matrice de contact Wt (dico)
#correspondance des numeros de noeuds aux indices 0 a n-1!
def contact(snapt,correspnodes,version):
    """
    Builds a dictionary to represent the weighted adjacency matrix at a given
    time.

    Parameters:
        snapt (list): List of contacts formatted as (date,(source,destination))
        correspnodes (dict): Dictionary of correspondences between node labels
        and their index in [0,n-1]
        version (int): Indicates whether to compute TempoRank (version 1) or
        out-TempoRank (version 2)

    Returns:
        wt
    """
    #snapt =
    #Dico plutot que matrice
    Print = PETSc.Sys.Print
    ot = {}
    wt = {}
    dicowt = {}
    Print("Debut Wt")
    for i in snapt:
        #graphe non orienté => symetrique
        #dicowt[(i,j)] : Nb de contacts entre i et j
        #wt[i] : out-strength de i
        if tuple((correspnodes[i[0]],correspnodes[i[1]])) not in dicowt.keys():
            dicowt[tuple((correspnodes[i[0]],correspnodes[i[1]]))]=1
        else:
            dicowt[tuple((correspnodes[i[0]],correspnodes[i[1]]))]+=1
        if correspnodes[i[0]] not in wt.keys():
            wt[correspnodes[i[0]]] = 1
            if version==2:
                ot[correspnodes[i[0]]] = 1
        else:
            wt[correspnodes[i[0]]] += 1

    return wt,ot,dicowt


#Remplissage de la matrice de transition Bt (matrice Petsc)
def transition(wt,dicowt,q,n,d,ot,version):
    Print = PETSc.Sys.Print
    #Sparse Bt pas symetrique
    Bt = PETSc.Mat()
    Ot = PETSc.Vec()
    Bt.create(PETSc.COMM_WORLD)
    Bt.setSizes([n,n])
    Bt.setType('dense')     #sparse
    Bt.setUp()
    Istart, Iend = Bt.getOwnershipRange()
    Print("Debut Bt")
    for K in range(Istart, Iend) :
        si = float(wt.get(K,0.0)) # Est-ce que K a des voisins au temps t?
        if si == 0.0:
            for L in range(n):
                if K==L:
                    Bt[K,L] = q+(1-q)*d/n+(1-q)*(1-d)
                else:
                    Bt[K,L] = (1-q)*d/n
        else:
            for L in range(n):
                if K==L:
                    Bt[K,L] = q+(1-q)*d/n
                else:
                    wtij = dicowt.get(tuple((K,L)),0.0)
                    if wtij != 0.0:
                        Bt[K,L] = (1-q)* (wtij*(1-d)/si + d/n)
                    else:
                        Bt[K,L] = (1-q)*d/n
    Bt.assemblyBegin()
    Bt.assemblyEnd()
    Print("Fin Bt")
    if version==2:
        Print("Debut Ot")
        Ot.create(PETSc.COMM_WORLD)
        Ot.setSizes(n)
        Ot.setUp()
        Istart, Iend = Ot.getOwnershipRange()
        for K in range(Istart,Iend):
            si = float(wt.get(K,0.0))
            if si != 0.0:
                Ot[K] = 1
            else:
                Ot[K] = 0
        Ot.assemblyBegin()
        Ot.assemblyEnd()
    return Bt,Ot


def listetransition(dicosnap,correspnodes,n,q,d,version):
    """
    Assembles a list of transition matrices from a dictionary of snapshots.

    Parameters:
        dicosnap (dict): Dictionary of snapshots
        correspnodes (dict): Dictionary of correspondences between node labels
        and their index in [0,n-1]
        n (int): Number of nodes in the network
        q (float): Laziness parameter
        d (float): Teleportation parameter
        version (int): Indicates whether to compute TempoRank (version 1) or
        out-TempoRank (version 2)

    Returns:
        listtrans (list): A list of transition matrices (as PETSc.Mat objects)
        listod (list): A list of adjacency matrices (as PETSC.Mat objects)
    """
    listtrans = list()
    listod = list()
    for t in sorted(dicosnap.keys()):
        wt,ot,dicowt = contact(dicosnap[t]["edgelist"],correspnodes,version)
        Bt,Ot = transition(wt,dicowt,q,n,d,ot,version)
        PETSc.COMM_WORLD.barrier()
        listtrans.append(Bt.copy())
        if version == 2:
            listod.append(Ot.copy())
        Bt.destroy()
        Ot.destroy()
    return listtrans,listod


####Version par paquets
#Multiplication par paquets de toutes les matrices de transitions pour 1 mois
def multparpaquet(listtrans):
    Print = PETSc.Sys.Print
    Print(str(len(listtrans)-1)+" Produits matriciel a faire")
    #Multiplication par paquets (2 paquets)
    nbpaq = 2
    taillepaq = len(listtrans)//nbpaq
    listPtp = list()
    cpt = 0
    for i in range(nbpaq):
        Ptppart = None
        for j in range(taillepaq):
            if j==0:
                Ptppart = listtrans[j+(i*taillepaq)]
            else:
                cpt += 1
                Print("Produit matriciel "+str(cpt))
                Ptppart = Ptppart.matMult(listtrans[j+(i*taillepaq)])

        if len(listtrans)%nbpaq != 0:
            #Cas 2 paquets
            if i==(nbpaq-1):
                cpt += 1
                Print("Produit matriciel "+str(cpt))
                Ptppart = Ptppart.matMult(listtrans[-1])

        listPtp.append(Ptppart)

    #Cas 2 paquets
    Print("Multiplication des paquets")
    Ptp = listPtp[0].matMult(listPtp[1])

    Print("Fin produit matriciel")
    Print(Ptp.getInfo())
    return Ptp

# Calcul du vecteur propre dominant de Ptp en utilisant SLEPc

def slep_powerit(Ptp,n):
    Ptp.transpose()

    E = SLEPc.EPS(); E.create()
    E.setOperators(Ptp)
    E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    E.setTolerances(tol=1e-12)
    E.setFromOptions()
    E.solve()

    Print=PETSc.Sys.Print

    its=E.getIterationNumber()
    Print("Number of iterations of the method: %d" % its)

    eps_type = E.getType()
    Print("Solution method: %s" % eps_type)

    nconv = E.getConverged()
    Print("Number of converged eigenpairs %d" % nconv)

    tol, maxit = E.getTolerances()
    Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

    if nconv >0:
        vpost, wpost = Ptp.getVecs()
        vposti, wposti = Ptp.getVecs()
        k=E.getEigenpair(0,vpost,vposti)
        Print("Valeur propre"+str(k))
        return abs(vpost)

#MANUAL POWER IT PETSC
def powerit(Ptp,n):
    #Vpre
    vpre = Ptp.getColumnVector(0)
    vpre.set(1/float(n))
    vpre.assemble()
    #Vpost
    vpost = Ptp.getColumnVector(0)
    vpost.set(0)
    vpost.assemble()
    #Vpost = vpre * Ptp
    Ptp.multTranspose(vpre,vpost)
    #error
    error_min = 10**-6
    error = math.sqrt(((vpost-vpre)*(vpost-vpre)).sum()/3)
    while error > error_min:
        vpre = vpost
        vpost = Ptp.getColumnVector(0)
        vpost.set(0)
        vpost.assemble()
        Ptp.multTranspose(vpre,vpost)
        error = math.sqrt(((vpost-vpre)*(vpost-vpre)).sum()/3)
    Print = PETSc.Sys.Print
    Print(error)
    vpre.destroy()
    return vpost



#Calcul du TR avec une liste de matrices de transitions
def temporankv2(v1,listtrans,listnodes,listod,version):
    vt = v1
    mask_vt = v1.copy()
    if version==2:
        ot = listod[0]
        mask_vt.pointwiseMult(ot,v1)
        PETSc.Sys.Print(mask_vt.getArray())
    sommev = mask_vt.copy()
    for t in range(len(listtrans)-1):
        vtplus1 = listtrans[t].getColumnVector(0).copy()
        vtplus1.set(0)
        vtplus1.assemble()
        listtrans[t].multTranspose(vt,vtplus1) # vtplus1 = P_t * vt
        if version==2:
            ot = listod[t+1]
            mask_vt.pointwiseMult(vtplus1,ot)
        else:
            mask_vt = vtplus1
        vt=vtplus1.copy()
        sommev = sommev + mask_vt
    meanv = sommev / len(listtrans)
    #Scatter
    comm = meanv.getComm()
    scat, V = PETSc.Scatter.toAll(meanv)
    scat.scatter(meanv, V, False, PETSc.Scatter.Mode.FORWARD)
    V=V/PETSc.Vec.sum(V)
    Print = PETSc.Sys.Print
    Print(V.getArray())
    TR = {}
    TR = {listnodes[i]: V[i] for i in range(len(listnodes))}

    return TR


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))

