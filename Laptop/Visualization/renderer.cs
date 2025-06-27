/**
 * @file renderer.c
 * @author Vipaswi Thapa (vt7637@g.rit.edu)
 * @brief A renderer that takes in quaternion data and converts it into euler angles for display using Unity.
 * @version 0.1
 * @date 2025-05-15
 * 
 * @copyright Copyright (c) 2025
 * 
 */

using System.Runtime.InteropServices;
using UnityEngine;

//needs to be under plugins/x86_64 in Unity 
[DllImport("UDPReceiver.dll", CallingConvention = CallingConvention.Cdecl)] //use C calling convention
private static extern System.IntPtr getLatestPacket(); //pointing to a motionPointer


// #region Structs
/*Mirrored structs from imu_packet.h:*/

[StructLayout(LayoutKind.Sequential)] //ensures that structures are defined sequentially in memory, which is required for InteropServices
public struct Quaternion {
  float qw, qx, qy, qz;
}

[StructLayout(LayoutKind.Sequential)] 
public struct IMUPacket {
  float ax, ay, az;
  float gx, gy, gz;
  float mx, my, mz;
}

[StructLayout(LayoutKind.Sequential)]
public struct motionPacket {
  IMUPacket rawData;
  Quaternion orientData;
}

// #endregion

public class renderer : MonoBehavior
{
  motionPacket latest;

  /**
    Called at the beginning
  **/
  void Start() { }

  /**
    Called once per frame
  **/
  void update()
  {
    IntPtr ptr = get_latest_packet();
    latest = Marshal.PtrToStructure<motionPacket>(ptr); //convert

    Quaternion q = latest.orientData;

    transform.rotation = new UnityEngine.Quaternion(q.qw, q.qx, q.qy, q.qz);
  }
}
 

 