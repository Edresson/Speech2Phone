
import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input( '../images/espectrogram.png',width=10, height=4,to='(0,2.2,0)' ),


    to_Dropout( "dp0", dp_rate="Dropout 20\%", offset="(0.6,0.5,0)", to="(0,0,0)", width=1.5, height=3, depth=25, caption=""),
    to_Conv("conv1", "7x7", 16, offset="(2,1,0)", to="(0,0,0)",width=1, height=20, depth=35, caption="Conv CReLU" ),

    to_connection("dp0","conv1"),
    to_Pool("maxpool1",zlabel='MaxPool 1x5', offset="(4,0,0)",width=1, height=15, depth=25, caption=""),
    to_connection("conv1","maxpool1"),


    to_Conv("conv2", "7x7", 16, offset="(6,0,0)", to="(0,0,0)",width=1, height=15, depth=25, caption="Conv CReLU" ),
    to_connection("maxpool1","conv2"),
    to_Pool("maxpool2",zlabel='MaxPool 1x2', offset="(8,0,0)",width=1, height=7, depth=30, caption=""),
    to_connection("conv2","maxpool2"),

    to_Conv("conv3", "7x7", 8, offset="(10,0,0)", to="(0,0,0)",width=1, height=7, depth=25, caption="Conv CReLU" ),
    to_connection("maxpool2","conv3"),
    to_Pool("maxpool3",zlabel='MaxPool 2x2', offset="(12,0,0)",width=1, height=4, depth=15, caption=""),
    to_connection("conv3","maxpool3"),

    to_Dropout( "dp1", dp_rate="BNorm", offset="(14,0.0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, caption="", Color='\LnormDropoutColor'),

    to_connection("maxpool3","dp1"),
    to_Dense("dense2", 40 ,offset="(16,0,0)", caption="FC LeakyReLU",depth=20,Color ="\FcEluColor" ),
    #to_SoftMax("soft2", 10 ,"(5,0,0)", caption="SoftMax"  ),
    to_connection("dp1","dense2"),
    to_Dense("dense3", 900 ,offset="(19,0,0)", caption="FC LeakyReLU",Color ="\FcEluColor",depth=40 ),
    to_connection("dense2","dense3"),


    to_Rnn("Rnn1", 128 ,offset="(21,0,0)", caption="RNN ReLU",Color ="\FcReluColor",depth=5,width=5, height=5 ),
    to_connection("dense3","Rnn1"),
    
    to_Rnn("Rnn2", 80 ,offset="(24,0,0)", caption="RNN ReLU",Color ="\FcReluColor",depth=5,width=5, height=5 ),
    to_connection("Rnn1","Rnn2"),

    to_Dense("dense4", 572 ,offset="(27,0,0)", caption="FC Linear",Color ="\FcLinearColor",depth=35 ),
    to_connection("Rnn2","dense4"),
    to_output( '../images/outpu-a.png',width=2, height=4,to='(27,1.1,-2)' ),
    #to_connection("dense1", "soft1"),  
    #to_connection("pool2", "soft1"),    
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()

