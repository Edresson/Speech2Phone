
import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input( '../images/Mfcc.png',width=10, height=4,to='(0,2.2,0)' ),


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

    to_Conv("conv4", "7x7", 1, offset="(14,0,0)", to="(0,0,0)",width=1, height=4, depth=12, caption="Conv CReLU" ),
    to_connection("maxpool3","conv4"),

    to_Dropout( "dp1", dp_rate="BNorm+Dropout 40\%", offset="(16,0.0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, caption="", Color='\LnormDropoutColor'),

    to_connection("conv4","dp1"),
    to_Dense("dense2", 200 ,offset="(18,0,0)", caption="FC ELU",depth=60,Color ="\FcEluColor" ),
    #to_SoftMax("soft2", 10 ,"(5,0,0)", caption="SoftMax"  ),
    to_connection("dp1","dense2"),
    to_Dropout( "dp2", dp_rate="Dropout 40\%", offset="(20,0.0,0)", to="(0,0,0)", width=1.5, height=3, depth=25),
    to_connection("dense2","dp2"),
    to_Dense("dense3", 20 ,offset="(22,0,0)", caption="FC SoftMax",Color ="\FcSoftmaxColor",depth=30 ),
    to_connection("dp2","dense3"),
    
    #to_connection("dense1", "soft1"),  
    #to_connection("pool2", "soft1"),    
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")
