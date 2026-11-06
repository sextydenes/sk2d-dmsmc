import argparse
import numpy as np
import torch

nfile=1
sklambda=0.022
kappa=0.27
explsigma=25
link=0.0
widthfakt=256



def create_parser(defhelp):
    parser=argparse.ArgumentParser(description=defhelp)
    return parser



def add_defargs(parser):
    parser.add_argument("-kappa",type=float,help="kappa",default=0.27)
    parser.add_argument("-sklambda",type=float,help="lambda",default=0.022)
    parser.add_argument("-nfile",type=int,help="filenames get an index .(nfile) at the end",default=1)
    parser.add_argument("-Nb",type=int,help="Nbatch",default=64)
    parser.add_argument("-dev",type=str,help="whichdev")
    parser.add_argument("-explsigma",type=float,help="explsigma",default=25.0)
    parser.add_argument("-link",type=float,help="linear drift of noise proce",default=0.0)
    parser.add_argument("-widthfakt",type=int,help="size of network",default=256)
#parser.add_argument("file_A",type=str,help="the name of file A")
#parser.add_argument("file_B",type=str,help="the name of file B")
#parser.add_argument("whattodo",type=str,help="de")
#parser.add_argument("-ca",type=int,help="the common column in file A")
#parser.add_argument("-cb",type=int,help="the common column in file B")
#parser.add_argument("-sm",action="store_true",help="match the common field as a string") 
    args=parser.parse_args()


    global nfile,sklambda,kappa,explsigma,link,widthfakt
    explsigma=args.explsigma 
    nfile=args.nfile
    link=args.link
    kappa=args.kappa
    sklambda=args.sklambda
    
    return args
    



def goft(t):
  return np.power(explsigma,t)

def gint(t):
  return (np.power(explsigma,2.0*t)-np.exp(-2.0*link*t))/(2.0*(link+np.log(explsigma)))

def gint_nolinforce(t):
  return 0.5/np.log(explsigma)*(np.power(explsigma,2.0*t)-1.0)


def tgint(t):
  return (torch.pow(explsigma,2.0*t)-torch.exp(-2.0*link*t))*gintfakt

def sqrttgint(t):
  return torch.sqrt((torch.pow(explsigma,2.0*t)-torch.exp(-2.0*link*t))*gintfakt)
