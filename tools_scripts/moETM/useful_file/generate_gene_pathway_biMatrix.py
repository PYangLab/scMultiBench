import pandas as pd

gene_f=open('gex_name.csv','r')## gene names used in the model training
msigdb_f=open('c7.all.v2022.1.Hs.symbols.gmt','r')#downloaded from MSigDB

## read gene names
genes=pd.read_csv(gene_f)
gene_names=genes['Unnamed: 0']## one column containing gene names

## read pathways
msigdb_lines=msigdb_f.readlines()

## create and append dataframe
pathway_gene_m=pd.DataFrame(columns=gene_names)
for l in msigdb_lines:
    ll=l.rstrip().split('\t')
    p=ll[0]#pathway
    p_genes=ll[2:]
    if len(p_genes) > 5 and len(p_genes) < 1000:
        g=[]
        for i in gene_names:
            if i in p_genes:
                g.append(1)
            else:
                g.append(0)
        if sum(g)!=0:#remove pathways that no genes are included (rowSum==0)
            pathway_gene_m.loc[p]=g   
        else:
            pass
    else:
        pass

## save results
pathway_gene_m.to_csv('pathway_by_gene_binary.csv',columns=gene_names)

