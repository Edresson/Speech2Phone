
import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input( '../images/concatenacao-embedding.png',width=10, height=4,to='(0,2.2,0)' ),

    to_Dropout( "dp0", dp_rate="Dropout 10\%", offset="(-0.2,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, caption=""),
    to_Dense("dense1", 80 ,offset="(2,0,0)", caption="FC ReLU",Color ="\FcReluColor",depth=50 ),#relu color is default
    to_connection("dp0","dense1"),

    to_Dropout( "l22", dp_rate="L2 0.001", offset="(4,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, caption=""),
    to_connection("dense1","l22"),
    
    to_Dense("dense2", 8 ,offset="(6,0,0)", caption="FC CReLU",depth=60,Color ="\FcCreluColor" ),
    to_connection("dense1","dense2"),
    to_Dropout( "l21", dp_rate="L2 0.001", offset="(8,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, caption=""),
    to_connection("dense2","l21"),

    #to_SoftMax("soft2", 10 ,"(5,0,0)", caption="SoftMax"  ),
    to_Dense("dense3", 2 ,offset="(10,0,0)", caption="FC SoftMax",Color ="\FcSoftmaxColor",depth=30 ),
    to_connection("dense2","dense3"),

    #to_output( '../images/outpu-a.png',width=2, height=4,to='(7,1.1,-2)' ),
    #to_connection("dense1", "soft1"),  
    #to_connection("pool2", "soft1"),    
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
