"use client";
// components/FancyThreeScene.tsx
import React, { useRef, useEffect } from "react";
import * as THREE from "three";

const FancyThreeScene: React.FC = () => {
  const mountRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!mountRef.current) return;
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    // シーン・カメラ・レンダラー
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    mountRef.current.appendChild(renderer.domElement);

    // パーティクルシステム（星空風）
    const particleCount = 10000;
    const particleGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      const x = (Math.random() - 0.5) * 20; // -10 ～ 10
      const y = (Math.random() - 0.5) * 20;
      const z = (Math.random() - 0.5) * 20;
      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
      colors[i * 3] = (x + 10) / 20;
      colors[i * 3 + 1] = (y + 10) / 20;
      colors[i * 3 + 2] = (z + 10) / 20;
    }
    particleGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3)
    );
    particleGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(colors, 3)
    );

    const particleMaterial = new THREE.PointsMaterial({
      size: 0.1,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    });

    const particleSystem = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particleSystem);

    // トーラスノットを中央に配置（おしゃれな対象物）
    const knotGeometry = new THREE.TorusKnotGeometry(1, 0.3, 100, 16);
    const knotMaterial = new THREE.MeshStandardMaterial({
      color: 0xff4081,
      roughness: 0.5,
      metalness: 0.5,
    });
    const knot = new THREE.Mesh(knotGeometry, knotMaterial);
    scene.add(knot);

    // 照明
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(5, 5, 5);
    scene.add(pointLight);

    let frame = 0;
    const animate = () => {
      requestAnimationFrame(animate);

      // パーティクルシステムのゆっくり回転
      particleSystem.rotation.y += 0.0005;
      particleSystem.rotation.x += 0.0003;

      // トーラスノットの回転（動きにおしゃれさを追加）
      knot.rotation.x += 0.01;
      knot.rotation.y += 0.015;

      // ゆるやかなパララックス効果
      camera.position.x = Math.sin(frame * 0.001) * 0.5;
      camera.position.y = Math.cos(frame * 0.001) * 0.5;
      camera.lookAt(scene.position);

      renderer.render(scene, camera);
      frame++;
    };
    animate();

    return () => {
      if (mountRef.current) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
      particleGeometry.dispose();
      particleMaterial.dispose();
      knotGeometry.dispose();
      knotMaterial.dispose();
    };
  }, []);

  // ここで、クラス "h-[600px]" として高さを変更。お好みで "h-screen" なども利用可能です。
  return <div ref={mountRef} className="w-full h-screen" />;
};

export default FancyThreeScene;
