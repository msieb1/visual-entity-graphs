import subprocess as sb
from os.path import join

def main(args):

python python/collect_demo_w_agent.py 2/putAIntoB_bowlenv1_16.bin
    sb.call(['python', 'python/collect_demo_w_agent', '--play', args.record, '--path', args.savename])

    sb.call(['python', 'python/compute_mrcnn_output.py', '-d', join('/home/msieb/projects/gps-lfd/demo_data', args.savename)]
    
    sb.call(['python', 'python/gps/gps_main.py', '-experiment', args.experiment])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="execute data collection from stored demo records, compute mrcnn features and execute PILQR")

    args = parser.parse_args()
    main(args)
