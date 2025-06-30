using Godot;
using System;
using System.Collections.Generic;

namespace Terrain;

[Tool]
[GlobalClass]
public partial class DrawTerrain : CompositorEffect
{
    [Export]
    public bool Regenerate { get; set; } = true;

    [ExportGroup("Mesh Settings")]
    [Export(PropertyHint.Range, "2,1000,1,or_greater")]
    public int SideLength { get; set; } = 200;
    [Export(PropertyHint.Range, "0.01,1.0,0.01,or_greater")]
    public float MeshScale { get; set; } = 1.0f;
    [Export]
    public bool Wireframe { get; set; } = false;

    [ExportGroup("Noise Settings")]
    [Export]
    public int NoiseSeed { get; set; } = 0;
    [Export(PropertyHint.Range, "0.1,400,0.1,or_greater")]
    public float Zoom { get; set; } = 100.0f;
    [Export]
    public Vector3 Offset { get; set; } = Vector3.Zero;
    [Export(PropertyHint.Range, "-180.0,180.0")]
    public float GradientRotation { get; set; } = 0.0f;
    [Export(PropertyHint.Range, "1,32")]
    public int OctaveCount { get; set; } = 10;

    [ExportGroup("Octave Settings")]
    [Export(PropertyHint.Range, "-180.0,180.0")]
    public float Rotation { get; set; } = 30.0f;
    [Export]
    public Vector2 AngularVariance { get; set; } = Vector2.Zero;
    [Export(PropertyHint.Range, "0.01,2.0")]
    public float InitialAmplitude { get; set; } = 0.5f;
    [Export(PropertyHint.Range, "0.01,1.0")]
    public float AmplitudeDecay { get; set; } = 0.45f;
    [Export(PropertyHint.Range, "0.01,3.0")]
    public float Lacunarity { get; set; } = 2.0f;
    [Export]
    public Vector2 FrequencyVariance { get; set; } = Vector2.Zero;
    [Export(PropertyHint.Range, "0.0,300.0,0.1,or_greater")]
    public float HeightScale { get; set; } = 50.0f;

    [ExportGroup("Material Settings")]
    [Export]
    public float SlopeDamping { get; set; } = 0.2f;
    [Export]
    public Vector2 SlopeThreshold { get; set; } = new(0.9f, 0.98f);
    [Export]
    public Color LowSlopeColor { get; set; } = new(0.83f, 0.88f, 0.94f);
    [Export]
    public Color HighSlopeColor { get; set; } = new(0.16f, 0.1f, 0.1f);

    [ExportGroup("Light Settings")]
    [Export]
    public Color AmbientLight { get; set; } = Colors.DimGray;

    public Transform3D Transform { get; set; }
    public DirectionalLight3D Light { get; set; }

    public RenderingDevice RenderingDevice { get; set; }
    public Rid PFramebuffer { get; set; }
    public int ChachedFramebufferFormat { get; set; }

    public Rid PRenderPipeline;
    public Rid PRenderPipelineUniformSet;
    public Rid PWireRenderPipeline;
    public Rid PVertexBuffer;
    public long VertexFormat;
    public Rid PVertexArray;
    public Rid PIndexBuffer;
    public Rid PIndexArray;
    public Rid PWireIndexBuffer;
    public Rid PWireIndexArray;
    public Rid PShader;
    public Rid PWireShader;
    public Color[] ClearColors = [Colors.DarkBlue];

    public DrawTerrain()
    {
        EffectCallbackType = EffectCallbackTypeEnum.PostTransparent;

        RenderingDevice = RenderingServer.GetRenderingDevice();

        SceneTree tree = (SceneTree)Engine.GetMainLoop();
        Node root = Engine.IsEditorHint() ? tree.EditedSceneRoot : tree.CurrentScene;

        if (root is not null) Light = root.GetNodeOrNull<DirectionalLight3D>("DirectionalLight3D");
    }

    public Rid CompileShader(string vertexShader, string fragmentShader)
    {
        var src = new RDShaderSource
        {
            SourceVertex = vertexShader,
            SourceFragment = fragmentShader
        };

        RDShaderSpirV shaderSpirV = RenderingDevice.ShaderCompileSpirVFromSource(src);

        if (!string.IsNullOrWhiteSpace(shaderSpirV.CompileErrorVertex)) GD.PushError(shaderSpirV.CompileErrorVertex);
        if (!string.IsNullOrWhiteSpace(shaderSpirV.CompileErrorFragment)) GD.PushError(shaderSpirV.CompileErrorFragment);

        return RenderingDevice.ShaderCreateFromSpirV(shaderSpirV);
    }

    public void InitializeRender(int framebufferFormat)
    {
        PShader = CompileShader(VERTEX_SOURCE, FRAGMENT_SOURCE);
        PWireShader = CompileShader(VERTEX_SOURCE, WIRE_FRAGMENT_SOURCE);

        List<float> vertexBuffer = [];
        var halfLength = (SideLength - 1) / 2.0f;

        for (int x = 0; x < SideLength; x++)
        {
            for (int z = 0; z < SideLength; z++)
            {
                Vector2 xz = new Vector2(x - halfLength, z - halfLength) * MeshScale;
                Vector3 pos = new(xz.X, 0, xz.Y);

                Vector4 color = new(GD.Randf(), GD.Randf(), GD.Randf(), 1f);

                vertexBuffer.AddRange([pos.X, pos.Y, pos.Z]);
                vertexBuffer.AddRange([color.X, color.Y, color.Z, color.W]);
            }
        }

        int vertexCount = vertexBuffer.Count / 7;
        GD.Print($"Vertex Coount: {vertexCount}");

        List<int> indexBuffer = [];
        List<int> wireIndexBuffer = [];

        for (int row = 0; row < SideLength * (SideLength - 1); row += SideLength)
        {
            for (int i = 0; i < SideLength - 1; i++)
            {
                int v = i + row;

                int v0 = v;
                int v1 = v + SideLength;
                int v2 = v + SideLength + 1;
                int v3 = v + 1;

                indexBuffer.AddRange([v0, v1, v3, v1, v2, v3]);
                wireIndexBuffer.AddRange([v0, v1, v0, v3, v1, v3, v1, v2, v2, v3]);
            }
        }

        GD.Print($"Triangle Count: {indexBuffer.Count / 3}");

        float[] vertexBufferArr = [.. vertexBuffer];
        byte[] vertexBufferBytes = new byte[vertexBufferArr.Length * sizeof(float)];
        Buffer.BlockCopy(vertexBufferArr, 0, vertexBufferBytes, 0, vertexBufferBytes.Length);
        PVertexBuffer = RenderingDevice.VertexBufferCreate((uint)vertexBufferBytes.Length, vertexBufferBytes);

        Godot.Collections.Array<Rid> vertexBuffers = [PVertexBuffer, PVertexBuffer];

        const int STRIDE = 7;

        Godot.Collections.Array<RDVertexAttribute> vertexAttrs = [
            new RDVertexAttribute()
            {
                Format = RenderingDevice.DataFormat.R32G32B32Sfloat,
                Location = 0,
                Offset = 0,
                Stride = STRIDE * sizeof(float)
            },
            new RDVertexAttribute()
            {
                Format = RenderingDevice.DataFormat.R32G32B32A32Sfloat,
                Location = 1,
                Offset = 3 * sizeof(float),
                Stride = STRIDE * sizeof(float)
            }
        ];

        VertexFormat = RenderingDevice.VertexFormatCreate(vertexAttrs);
        PVertexArray = RenderingDevice.VertexArrayCreate((uint)(vertexBuffer.Count / STRIDE), VertexFormat, vertexBuffers);

        int[] indexBufferArr = [.. indexBuffer];
        byte[] indexBufferBytes = new byte[indexBufferArr.Length * sizeof(int)];
        Buffer.BlockCopy(indexBufferArr, 0, indexBufferBytes, 0, indexBufferBytes.Length);
        PIndexBuffer = RenderingDevice.IndexBufferCreate((uint)indexBufferArr.Length, RenderingDevice.IndexBufferFormat.Uint32, indexBufferBytes);

        int[] wireIndexBufferArr = [.. wireIndexBuffer];
        byte[] wireIndexBufferBytes = new byte[wireIndexBufferArr.Length * sizeof(int)];
        Buffer.BlockCopy(wireIndexBufferArr, 0, wireIndexBufferBytes, 0, wireIndexBufferBytes.Length);
        PWireIndexBuffer = RenderingDevice.IndexBufferCreate((uint)wireIndexBufferArr.Length, RenderingDevice.IndexBufferFormat.Uint32, wireIndexBufferBytes);

        PIndexArray = RenderingDevice.IndexArrayCreate(PIndexBuffer, 0, (uint)indexBuffer.Count);
        PWireIndexArray = RenderingDevice.IndexArrayCreate(PWireIndexBuffer, 0, (uint)wireIndexBuffer.Count);

        InitializeRenderPipelines(framebufferFormat);
    }

    public void InitializeRenderPipelines(int framebufferFormat)
    {
        var rasterState = new RDPipelineRasterizationState()
        {
            CullMode = RenderingDevice.PolygonCullMode.Back
        };

        var depthState = new RDPipelineDepthStencilState
        {
            EnableDepthWrite = true,
            EnableDepthTest = true,
            DepthCompareOperator = RenderingDevice.CompareOperator.Greater
        };

        var blend = new RDPipelineColorBlendState();
        blend.Attachments.Add(new RDPipelineColorBlendStateAttachment());

        PRenderPipeline = RenderingDevice.RenderPipelineCreate(
            PShader,
            framebufferFormat,
            VertexFormat,
            RenderingDevice.RenderPrimitive.Triangles,
            rasterState,
            new RDPipelineMultisampleState(),
            depthState,
            blend
        );

        PWireRenderPipeline = RenderingDevice.RenderPipelineCreate(
            PWireShader,
            framebufferFormat,
            VertexFormat,
            RenderingDevice.RenderPrimitive.Lines,
            rasterState,
            new RDPipelineMultisampleState(),
            depthState,
            blend
        );
    }

    public override void _RenderCallback(int effectCallbackType, RenderData renderData)
    {
        if (!Enabled) return;
        if (effectCallbackType != (int)EffectCallbackType) return;

        var sceneData = renderData.GetRenderSceneData();

        if (renderData.GetRenderSceneBuffers() is not RenderSceneBuffersRD sceneBuffers) return;

        if (Regenerate || !PRenderPipeline.IsValid)
        {
            _Notification((int)NotificationPredelete);
            PFramebuffer = FramebufferCacheRD.GetCacheMultipass([sceneBuffers.GetColorTexture(), sceneBuffers.GetDepthTexture()], [], 1);
            InitializeRender((int)RenderingDevice.FramebufferGetFormat(PFramebuffer));
            Regenerate = false;
        }

        var currentFramebuffer = FramebufferCacheRD.GetCacheMultipass([sceneBuffers.GetColorTexture(), sceneBuffers.GetDepthTexture()], [], 1);

        if (PFramebuffer != currentFramebuffer)
        {
            PFramebuffer = currentFramebuffer;
            InitializeRenderPipelines((int)RenderingDevice.FramebufferGetFormat(PFramebuffer));
        }

        var model = Transform;
        var view = sceneData.GetCamTransform().Inverse();
        var projection = sceneData.GetViewProjection(0);

        var mvp = projection * new Projection(view * model);

        List<float> buffer = [];
        for (int i = 0; i < 16; i++)
        {
            buffer.Add(mvp[i / 4][i % 4]);
        }

        Vector3 lightDirection = new(0f, 1f, 0f);

        if (Light is null)
        {
            var tree = Engine.GetMainLoop() as SceneTree;
            var root = Engine.IsEditorHint() ? tree?.EditedSceneRoot : tree?.CurrentScene;
            Light = root?.GetNodeOrNull<DirectionalLight3D>("DirectionalLight3D");
            if (Light is null) GD.PushError("No DirectionalLight3D found in scene");
        }
        else
        {
            lightDirection = Light.Transform.Basis.Z.Normalized();
        }

        buffer.AddRange([
            lightDirection.X,
            lightDirection.Y,
            lightDirection.Z,
            GradientRotation,
            Rotation,
            HeightScale,
            AngularVariance.X,
            AngularVariance.Y,
            Zoom,
            OctaveCount,
            AmplitudeDecay,
            1.0f,
            Offset.X,
            Offset.Y,
            Offset.Z,
            NoiseSeed,
            InitialAmplitude,
            Lacunarity,
            SlopeThreshold.X,
            SlopeThreshold.Y,
            LowSlopeColor.R,
            LowSlopeColor.G,
            LowSlopeColor.B,
            1.0f,
            HighSlopeColor.R,
            HighSlopeColor.G,
            HighSlopeColor.B,
            1.0f,
            FrequencyVariance.X,
            FrequencyVariance.Y,
            SlopeDamping,
            1.0f,
            AmbientLight.R,
            AmbientLight.G,
            AmbientLight.B,
            1.0f
        ]);

        float[] bufferArr = [.. buffer];
        byte[] bufferBytes = new byte[bufferArr.Length * sizeof(float)];
        Buffer.BlockCopy(bufferArr, 0, bufferBytes, 0, bufferBytes.Length);

        var pUniformBuf = RenderingDevice.UniformBufferCreate((uint)bufferBytes.Length, bufferBytes);

        var uniform = new RDUniform
        {
            Binding = 0,
            UniformType = RenderingDevice.UniformType.UniformBuffer
        };

        uniform.AddId(pUniformBuf);

        if (PRenderPipelineUniformSet.IsValid) RenderingDevice.FreeRid(PRenderPipelineUniformSet);

        PRenderPipelineUniformSet = RenderingDevice.UniformSetCreate([uniform], PShader, 0);

        RenderingDevice.DrawCommandBeginLabel("Terrain Mesh", new(1f, 1f, 1f, 1f));

        var drawList = RenderingDevice.DrawListBegin(PFramebuffer, RenderingDevice.DrawFlags.IgnoreAll, ClearColors, 1.0f, 0, new(), 0);

        RenderingDevice.DrawListBindRenderPipeline(drawList, Wireframe ? PWireRenderPipeline : PRenderPipeline);
        RenderingDevice.DrawListBindVertexArray(drawList, PVertexArray);
        RenderingDevice.DrawListBindIndexArray(drawList, Wireframe ? PWireIndexArray : PIndexArray);
        RenderingDevice.DrawListBindUniformSet(drawList, PRenderPipelineUniformSet, 0);
        RenderingDevice.DrawListDraw(drawList, true, 1);
        RenderingDevice.DrawListEnd();

        RenderingDevice.DrawCommandEndLabel();
    }

    public override void _Notification(int what)
    {
        if (what == (int)NotificationPredelete)
        {
            if (PRenderPipeline.IsValid) RenderingDevice.FreeRid(PRenderPipeline);
            if (PWireRenderPipeline.IsValid) RenderingDevice.FreeRid(PWireRenderPipeline);
            if (PVertexArray.IsValid) RenderingDevice.FreeRid(PVertexArray);
            if (PVertexBuffer.IsValid) RenderingDevice.FreeRid(PVertexBuffer);
            if (PIndexArray.IsValid) RenderingDevice.FreeRid(PIndexArray);
            if (PIndexBuffer.IsValid) RenderingDevice.FreeRid(PIndexBuffer);
            if (PWireIndexArray.IsValid) RenderingDevice.FreeRid(PWireIndexArray);
            if (PWireIndexBuffer.IsValid) RenderingDevice.FreeRid(PWireIndexBuffer);
        }
    }

    private const string VERTEX_SOURCE = """
        #version 450

        // This is the uniform buffer that contains all of the settings we sent over from the cpu in _render_callback. Must match with the one in the fragment shader.
        layout(set = 0, binding = 0, std140) uniform UniformBufferObject {
            mat4 MVP;
            vec3 _LightDirection;
            float _GradientRotation;
            float _NoiseRotation;
            float _TerrainHeight;
            vec2 _AngularVariance;
            float _Scale;
            float _Octaves;
            float _AmplitudeDecay;
            float _NormalStrength;
            vec3 _Offset;
            float _Seed;
            float _InitialAmplitude;
            float _Lacunarity;
            vec2 _SlopeRange;
            vec4 _LowSlopeColor;
            vec4 _HighSlopeColor;
            float _FrequencyVarianceLowerBound;
            float _FrequencyVarianceUpperBound;
            float _SlopeDamping;
            vec4 _AmbientLight;
        };
        
        // This is the vertex data layout that we defined in initialize_render after line 198
        layout(location = 0) in vec3 a_Position;
        layout(location = 1) in vec4 a_Color;

        // This is what the vertex shader will output and send to the fragment shader.
        layout(location = 2) out vec4 v_Color;
        layout(location = 3) out vec3 pos;

        #define PI 3.141592653589793238462
        
        // UE4's PseudoRandom function
        // https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Shaders/Private/Random.ush
        float pseudo(vec2 v) {
            v = fract(v/128.)*128. + vec2(-64.340622, -72.465622);
            return fract(dot(v.xyx * v.xyy, vec3(20.390625, 60.703125, 2.4281209)));
        }

        // Takes our xz positions and turns them into a random number between 0 and 1 using the above pseudo random function
        float HashPosition(vec2 pos) {
            return pseudo(pos * vec2(_Seed, _Seed + 4));
        }

        // Generates a random gradient vector for the perlin noise lattice points, watch my perlin noise video for a more in depth explanation
        vec2 RandVector(float seed) {
            float theta = seed * 360 * 2 - 360;
            theta += _GradientRotation;
            theta = theta * PI / 180.0;
            return normalize(vec2(cos(theta), sin(theta)));
        }

        // Normal smoothstep is cubic -- to avoid discontinuities in the gradient, we use a quintic interpolation instead as explained in my perlin noise video
        vec2 quinticInterpolation(vec2 t) {
            return t * t * t * (t * (t * vec2(6) - vec2(15)) + vec2(10));
        }

        // Derivative of above function
        vec2 quinticDerivative(vec2 t) {
            return vec2(30) * t * t * (t * (t - vec2(2)) + vec2(1));
        }

        // it's perlin noise that returns the noise in the x component and the derivatives in the yz components as explained in my perlin noise video
        vec3 perlin_noise2D(vec2 pos) {
            vec2 latticeMin = floor(pos);
            vec2 latticeMax = ceil(pos);

            vec2 remainder = fract(pos);

            // Lattice Corners
            vec2 c00 = latticeMin;
            vec2 c10 = vec2(latticeMax.x, latticeMin.y);
            vec2 c01 = vec2(latticeMin.x, latticeMax.y);
            vec2 c11 = latticeMax;

            // Gradient Vectors assigned to each corner
            vec2 g00 = RandVector(HashPosition(c00));
            vec2 g10 = RandVector(HashPosition(c10));
            vec2 g01 = RandVector(HashPosition(c01));
            vec2 g11 = RandVector(HashPosition(c11));

            // Directions to position from lattice corners
            vec2 p0 = remainder;
            vec2 p1 = p0 - vec2(1.0);

            vec2 p00 = p0;
            vec2 p10 = vec2(p1.x, p0.y);
            vec2 p01 = vec2(p0.x, p1.y);
            vec2 p11 = p1;
            
            vec2 u = quinticInterpolation(remainder);
            vec2 du = quinticDerivative(remainder);

            float a = dot(g00, p00);
            float b = dot(g10, p10);
            float c = dot(g01, p01);
            float d = dot(g11, p11);

            // Expanded interpolation freaks of nature from https://iquilezles.org/articles/gradientnoise/
            float noise = a + u.x * (b - a) + u.y * (c - a) + u.x * u.y * (a - b - c + d);

            vec2 gradient = g00 + u.x * (g10 - g00) + u.y * (g01 - g00) + u.x * u.y * (g00 - g10 - g01 + g11) + du * (u.yx * (a - b - c + d) + vec2(b, c) - a);
            return vec3(noise, gradient);
        }

        // The fractional brownian motion that sums many noise values as explained in the video accompanying this project
        vec3 fbm(vec2 pos) {
            float lacunarity = _Lacunarity;
            float amplitude = _InitialAmplitude;

            // height sum
            float height = 0.0;

            // derivative sum
            vec2 grad = vec2(0.0);

            // accumulated rotations
            mat2 m = mat2(1.0, 0.0,
                          0.0, 1.0);

            // generate random angle variance if applicable
            float angle_variance = mix(_AngularVariance.x, _AngularVariance.y, HashPosition(vec2(_Seed, 827)));
            float theta = (_NoiseRotation + angle_variance) * PI / 180.0;

            // rotation matrix
            mat2 m2 = mat2(cos(theta), -sin(theta),
                             sin(theta),  cos(theta));
                
            mat2 m2i = inverse(m2);

            for(int i = 0; i < int(_Octaves); ++i) {
                vec3 n = perlin_noise2D(pos);
                
                // add height scaled by current amplitude
                height += amplitude * n.x;    
                
                // add gradient scaled by amplitude and transformed by accumulated rotations
                grad += amplitude * m * n.yz;
                
                // apply amplitude decay to reduce impact of next noise layer
                amplitude *= _AmplitudeDecay;
                
                // generate random angle variance if applicable
                angle_variance = mix(_AngularVariance.x, _AngularVariance.y, HashPosition(vec2(i * 419, _Seed)));
                theta = (_NoiseRotation + angle_variance) * PI / 180.0;

                // reconstruct rotation matrix, kind of a performance stink since this is technically expensive and doesn't need to be done if no random angle variance but whatever it's 2025
                m2 = mat2(cos(theta), -sin(theta),
                            sin(theta),  cos(theta));
                
                m2i = inverse(m2);

                // generate frequency variance if applicable
                float freq_variance = mix(_FrequencyVarianceLowerBound, _FrequencyVarianceUpperBound, HashPosition(vec2(i * 422, _Seed)));

                // apply frequency adjustment to sample position for next noise layer
                pos = (lacunarity + freq_variance) * m2 * pos;
                m = (lacunarity + freq_variance) * m2i * m;
            }

            return vec3(height, grad);
        }
        
        void main() {
            // Passes the vertex color over to the fragment shader, even though we don't use it but you can use it if you want I guess
            v_Color = a_Color;

            // The fragment shader also calculates the fractional brownian motion for pixel perfect normal vectors and lighting, so we pass the vertex position to the fragment shader
            pos = a_Position;

            // Initial noise sample position offset and scaled by uniform variables
            vec3 noise_pos = (pos + vec3(_Offset.x, 0, _Offset.z)) / _Scale;

            // The fractional brownian motion
            vec3 n = fbm(noise_pos.xz);

            // Adjust height of the vertex by fbm result scaled by final desired amplitude
            pos.y += _TerrainHeight * n.x + _TerrainHeight - _Offset.y;
            
            // Multiply final vertex position with model/view/projection matrices to convert to clip space
            gl_Position = MVP * vec4(pos, 1);
        }
    """;

    private const string FRAGMENT_SOURCE = """
        #version 450

        // This is the uniform buffer that contains all of the settings we sent over from the cpu in _render_callback. Must match with the one in the vertex shader, they're technically the same thing occupying the same spot in memory this is just duplicate code required for compilation.
        layout(set = 0, binding = 0, std140) uniform UniformBufferObject {
            mat4 MVP;
            vec3 _LightDirection;
            float _GradientRotation;
            float _NoiseRotation;
            float _TerrainHeight;
            vec2 _AngularVariance;
            float _Scale;
            float _Octaves;
            float _AmplitudeDecay;
            float _NormalStrength;
            vec3 _Offset;
            float _Seed;
            float _InitialAmplitude;
            float _Lacunarity;
            vec2 _SlopeRange;
            vec4 _LowSlopeColor;
            vec4 _HighSlopeColor;
            float _FrequencyVarianceLowerBound;
            float _FrequencyVarianceUpperBound;
            float _SlopeDamping;
            vec4 _AmbientLight;
        };
        
        // These are the variables that we expect to receive from the vertex shader
        layout(location = 2) in vec4 a_Color;
        layout(location = 3) in vec3 pos;
        
        // This is what the fragment shader will output, usually just a pixel color
        layout(location = 0) out vec4 frag_color;

        #define PI 3.141592653589793238462
        
        // UE4's PseudoRandom function
        // https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Shaders/Private/Random.ush
        float pseudo(vec2 v) {
            v = fract(v/128.)*128. + vec2(-64.340622, -72.465622);
            return fract(dot(v.xyx * v.xyy, vec3(20.390625, 60.703125, 2.4281209)));
        }

        // Takes our xz positions and turns them into a random number between 0 and 1 using the above pseudo random function
        float HashPosition(vec2 pos) {
            return pseudo(pos * vec2(_Seed, _Seed + 4));
        }

        // Generates a random gradient vector for the perlin noise lattice points, watch my perlin noise video for a more in depth explanation
        vec2 RandVector(float seed) {
            float theta = seed * 360 * 2 - 360;
            theta += _GradientRotation;
            theta = theta * PI / 180.0;
            return normalize(vec2(cos(theta), sin(theta)));
        }

        // Normal smoothstep is cubic -- to avoid discontinuities in the gradient, we use a quintic interpolation instead as explained in my perlin noise video
        vec2 quinticInterpolation(vec2 t) {
            return t * t * t * (t * (t * vec2(6) - vec2(15)) + vec2(10));
        }

        // Derivative of above function
        vec2 quinticDerivative(vec2 t) {
            return vec2(30) * t * t * (t * (t - vec2(2)) + vec2(1));
        }

        // it's perlin noise that returns the noise in the x component and the derivatives in the yz components as explained in my perlin noise video
        vec3 perlin_noise2D(vec2 pos) {
            vec2 latticeMin = floor(pos);
            vec2 latticeMax = ceil(pos);

            vec2 remainder = fract(pos);

            // Lattice Corners
            vec2 c00 = latticeMin;
            vec2 c10 = vec2(latticeMax.x, latticeMin.y);
            vec2 c01 = vec2(latticeMin.x, latticeMax.y);
            vec2 c11 = latticeMax;

            // Gradient Vectors assigned to each corner
            vec2 g00 = RandVector(HashPosition(c00));
            vec2 g10 = RandVector(HashPosition(c10));
            vec2 g01 = RandVector(HashPosition(c01));
            vec2 g11 = RandVector(HashPosition(c11));

            // Directions to position from lattice corners
            vec2 p0 = remainder;
            vec2 p1 = p0 - vec2(1.0);

            vec2 p00 = p0;
            vec2 p10 = vec2(p1.x, p0.y);
            vec2 p01 = vec2(p0.x, p1.y);
            vec2 p11 = p1;
            
            vec2 u = quinticInterpolation(remainder);
            vec2 du = quinticDerivative(remainder);

            float a = dot(g00, p00);
            float b = dot(g10, p10);
            float c = dot(g01, p01);
            float d = dot(g11, p11);

            // Expanded interpolation freaks of nature from https://iquilezles.org/articles/gradientnoise/
            float noise = a + u.x * (b - a) + u.y * (c - a) + u.x * u.y * (a - b - c + d);

            vec2 gradient = g00 + u.x * (g10 - g00) + u.y * (g01 - g00) + u.x * u.y * (g00 - g10 - g01 + g11) + du * (u.yx * (a - b - c + d) + vec2(b, c) - a);
            return vec3(noise, gradient);
        }

        // The fractional brownian motion that sums many noise values as explained in the video accompanying this project
        vec3 fbm(vec2 pos) {
            float lacunarity = _Lacunarity;
            float amplitude = _InitialAmplitude;

            // height sum
            float height = 0.0;

            // derivative sum
            vec2 grad = vec2(0.0);

            // accumulated rotations
            mat2 m = mat2(1.0, 0.0,
                          0.0, 1.0);

            // generate random angle variance if applicable
            float angle_variance = mix(_AngularVariance.x, _AngularVariance.y, HashPosition(vec2(_Seed, 827)));
            float theta = (_NoiseRotation + angle_variance) * PI / 180.0;

            // rotation matrix
            mat2 m2 = mat2(cos(theta), -sin(theta),
                             sin(theta),  cos(theta));
                
            mat2 m2i = inverse(m2);

            for(int i = 0; i < int(_Octaves); ++i) {
                vec3 n = perlin_noise2D(pos);
                
                // add height scaled by current amplitude
                height += amplitude * n.x;    
                
                // add gradient scaled by amplitude and transformed by accumulated rotations
                grad += amplitude * m * n.yz;
                
                // apply amplitude decay to reduce impact of next noise layer
                amplitude *= _AmplitudeDecay;
                
                // generate random angle variance if applicable
                angle_variance = mix(_AngularVariance.x, _AngularVariance.y, HashPosition(vec2(i * 419, _Seed)));
                theta = (_NoiseRotation + angle_variance) * PI / 180.0;

                // reconstruct rotation matrix, kind of a performance stink since this is technically expensive and doesn't need to be done if no random angle variance but whatever it's 2025
                m2 = mat2(cos(theta), -sin(theta),
                            sin(theta),  cos(theta));
                
                m2i = inverse(m2);

                // generate frequency variance if applicable
                float freq_variance = mix(_FrequencyVarianceLowerBound, _FrequencyVarianceUpperBound, HashPosition(vec2(i * 422, _Seed)));

                // apply frequency adjustment to sample position for next noise layer
                pos = (lacunarity + freq_variance) * m2 * pos;
                m = (lacunarity + freq_variance) * m2i * m;
            }

            return vec3(height, grad);
        }
        
        void main() {
            // Recalculate initial noise sampling position same as vertex shader
            vec3 noise_pos = (pos + vec3(_Offset.x, 0, _Offset.z)) / _Scale;

            // Calculate fbm, we don't care about the height just the derivatives here for the normal vector so the ` + _TerrainHeight - _Offset.y` drops off as it isn't relevant to the derivative
            vec3 n = _TerrainHeight * fbm(noise_pos.xz);

            // To more easily customize the color slope blending this is a separate normal vector with its horizontal gradients significantly reduced so the normal points upwards more
            vec3 slope_normal = normalize(vec3(-n.y, 1, -n.z) * vec3(_SlopeDamping, 1, _SlopeDamping));

            // Use the slope of the above normal to create the blend value between the two terrain colors
            float material_blend_factor = smoothstep(_SlopeRange.x, _SlopeRange.y, 1 - slope_normal.y);

            // Blend between the two terrain colors
            vec4 albedo = mix(_LowSlopeColor, _HighSlopeColor, vec4(material_blend_factor));

            // This is the actual surface normal vector
            vec3 normal = normalize(vec3(-n.y, 1, -n.z));

            // Lambertian diffuse, negative dot product values clamped off because negative light doesn't exist
            float ndotl = clamp(dot(_LightDirection, normal), 0, 1);

            // Direct light cares about the diffuse result, ambient light does not
            vec4 direct_light = albedo * ndotl;
            vec4 ambient_light = albedo * _AmbientLight;

            // Combine lighting values, clip to prevent pixel values greater than 1 which would really really mess up the gamma correction below
            vec4 lit = clamp(direct_light + ambient_light, vec4(0), vec4(1));

            // Convert from linear rgb to srgb for proper color output, ideally you'd do this as some final post processing effect because otherwise you will need to revert this gamma correction elsewhere
            frag_color = pow(lit, vec4(2.2));
        }
    """;

    private const string WIRE_FRAGMENT_SOURCE = """
        #version 450

        layout(set = 0, binding = 0, std140) uniform UniformBufferObject {
            mat4 MVP; // 64 -> 0
            vec3 _LightDirection; // 16 -> 64
            float _GradientRotation;
            float _NoiseRotation; // 4 -> 80
            float _Amplitude; // 4 -> 84
            vec2 _AngularVariance; // 8 -> 88
            float _Frequency; // 4 -> 96
            float _Octaves; // 4 -> 100
            float _AmplitudeDecay; // 4 -> 104
            float _NormalStrength; // 4  -> 108
            vec3 _Offset; // 16 -> 112 -> 128
            float _Seed;
            float _InitialAmplitude;
            float _Lacunarity;
            vec2 _SlopeRange;
            vec4 _LowSlopeColor;
            vec4 _HighSlopeColor;
            float _FrequencyVarianceLowerBound;
            float _FrequencyVarianceUpperBound;
            float _SlopeDamping;
            vec4 _AmbientLight;
        };
        
        layout(location = 2) in vec4 a_Color;
        
        layout(location = 0) out vec4 frag_color;
        
        void main(){
            frag_color = vec4(1, 0, 0, 1);
        }
    """;
}
