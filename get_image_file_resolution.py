from PIL import Image
import sys
import tifffile
import xml.etree.ElementTree as ET


def get_resolution_not_ome(f_name):
    f_img = Image.open(f_name)
    res = f_img.info.get('resolution', None)
    res = [float(r) for r in res]
    aspect_fraction = (res[0] - res[1]) / res[0]
    assert aspect_fraction < 0.03
    return res[0]


def get_resolution(f_name):
    """ returns pixels per micron """
    with tifffile.TiffFile(f_name) as tif:
    # Access the OME-XML metadata
        ome_xml = tif.ome_metadata
        
    # short circuit
    if ome_xml is None:
        return get_resolution_not_ome(f_name)
        
    root = ET.fromstring(ome_xml)

    # Define the namespace to search for elements
    namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

    # Find the Pixels element and then extract the PhysicalSizeX attribute
    physical_size_x = root.find('.//ome:Pixels', namespaces).get('PhysicalSizeX')
    return float(physical_size_x)


def get_nuc_diameter(f_name):
    """ Return the pixel nuclei diameter for direct use in Cellpose methods """
    res = get_resolution(f_name)
    nuc_diam = res * 6
    return nuc_diam


def get_fiber_diameter(f_name):
    """ Return the pixel fiber diameter for direct use in Cellpose methods """
    res = get_resolution(f_name)
    nuc_diam = res * 60
    return nuc_diam


if __name__ == "__main__":
    f_name = sys.argv[1]
    res = get_resolution(f_name)
    sys.stdout.write(str(res)+'\n')
    sys.exit(0)

