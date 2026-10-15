// Builds HiGHS 1.8.0 (src/highs/, MIT; see THIRD_PARTY_NOTICES.md) when the `highs` feature is on:
// MWPF's relaxer optimizer (src/mwpf/relaxer_optimizer.rs) solves its linear programs with it, as
// MWPF's wheels do, so that the relaxers, and so the subgraphs and bounds, are MWPF's. The flags
// are those of HiGHS's own CMake Release build (FAST_BUILD, as the `highs-sys` crate builds it),
// so that it computes as MWPF's copy does. Without the feature nothing is built.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "highs")]
    highs::build();
}

#[cfg(feature = "highs")]
mod highs {
    use std::fs;
    use std::path::{Path, PathBuf};

    /// Every C, C++ (`.cpp`, `.cc`) source under `dir`, sorted.
    fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
        let mut entries: Vec<PathBuf> = fs::read_dir(dir).expect("src/highs").map(|entry| entry.unwrap().path()).collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                sources(&path, out);
            } else if matches!(path.extension().and_then(|e| e.to_str()), Some("c" | "cc" | "cpp")) {
                out.push(path);
            }
        }
    }

    pub fn build() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/highs");
        println!("cargo:rerun-if-changed={}", root.display());
        let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("highs_config");
        fs::create_dir_all(&out).unwrap();
        let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap();
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
        let msvc = target_env == "msvc";
        let x86 = target_arch == "x86_64" || target_arch == "x86";

        // HConfig.h as HiGHS's CMake writes it (src/HConfig.h.in): no zlib, 32-bit HighsInt, threads
        // on, and the intrinsics its configure step finds (`_mm_pause` on x86, `_BitScanReverse`
        // with MSVC, `__builtin_clz` otherwise)
        let mut config = String::from("#ifndef HCONFIG_H_\n#define HCONFIG_H_\n\n#define FAST_BUILD\n#define CMAKE_BUILD_TYPE \"Release\"\n");
        if x86 {
            config.push_str("#define HIGHS_HAVE_MM_PAUSE\n");
        }
        config.push_str(if msvc { "#define HIGHS_HAVE_BITSCAN_REVERSE\n" } else { "#define HIGHS_HAVE_BUILTIN_CLZ\n" });
        config.push_str("\n#define HIGHS_GITHASH \"n/a\"\n#define HIGHS_VERSION_MAJOR 1\n#define HIGHS_VERSION_MINOR 8\n#define HIGHS_VERSION_PATCH 0\n#define HIGHS_DIR \"\"\n\n#endif /* HCONFIG_H_ */\n");
        fs::write(out.join("HConfig.h"), config).unwrap();

        let mut files = vec![];
        sources(&root.join("src"), &mut files);
        sources(&root.join("extern"), &mut files);
        let (c_files, cpp_files): (Vec<PathBuf>, Vec<PathBuf>) = files.into_iter().partition(|path| path.extension().unwrap() == "c");

        // HiGHS's include directories (cmake/sources.cmake)
        let includes = [
            "extern", "extern/filereader", "extern/pdqsort", "extern/zstr", "src", "src/interfaces", "src/io", "src/ipm", "src/ipm/ipx", "src/ipm/basiclu", "src/lp_data", "src/mip", "src/model", "src/parallel", "src/pdlp", "src/pdlp/cupdlp", "src/presolve", "src/qpsolver", "src/simplex", "src/test", "src/util",
        ];
        let configure = |build: &mut cc::Build| {
            build.include(&out).opt_level(3).define("NDEBUG", None).warnings(false).cargo_warnings(false);
            for dir in includes {
                build.include(root.join(dir));
            }
            if msvc {
                // HiGHS's MSVC options (src/CMakeLists.txt)
                for flag in ["/bigobj", "/Zc:preprocessor", "/fp:precise"] {
                    build.flag(flag);
                }
                for define in ["NOMINMAX", "_CRT_SECURE_NO_WARNINGS", "_CRT_SECURE_NO_DEPRECATE"] {
                    build.define(define, None);
                }
                build.define("WIN32_LEAN_AND_MEAN", "1");
            } else if x86 && target_os != "windows" {
                build.flag("-mpopcnt");
            }
        };

        let mut cpp = cc::Build::new();
        cpp.cpp(true).files(&cpp_files);
        configure(&mut cpp);
        if !msvc {
            cpp.flag("-std=c++11");
        }
        cpp.compile("highs");

        let mut c = cc::Build::new();
        c.files(&c_files);
        configure(&mut c);
        c.compile("highs_c");

        // HiGHS runs its scheduler's worker threads
        if target_os == "linux" {
            println!("cargo:rustc-link-lib=pthread");
        }
    }
}
